# -*- coding: utf-8 -*-
"""
屏幕采集与显示器/窗口映射工具（macOS）

- 用 mss 抓取指定"原始像素"区域（自动处理 Retina 多屏缩放）。
- 用 pyobjc(Quartz/AppKit) 枚举正在运行的应用及其主窗口位置，
  实现"选择应用 -> 自动抓取它实时界面"。
- 若 pyobjc 不可用，应用自动抓取降级为手动框选，不影响其余功能。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import mss
    _MSS_OK = True
except Exception:  # pragma: no cover
    _MSS_OK = False

try:
    from Quartz import (
        CGGetOnlineDisplayList, CGDisplayBounds,
        CGDisplayPixelsWide, CGDisplayPixelsHigh,
        CGWindowListCopyWindowInfo, kCGWindowOwnerPID,
        kCGWindowBounds, kCGWindowName, kCGWindowLayer,
    )
    from Cocoa import NSWorkspace
    _QUARTZ_OK = True
except Exception:  # pragma: no cover
    _QUARTZ_OK = False


def _bounds_to_rect(b):
    """把窗口 bounds 解析成 (x, y, w, h)。

    macOS 的 `kCGWindowBounds` 在不同系统/pyobjc 绑定下可能是：
      - CGRect/NSRect 结构（有 .origin / .size 属性）
      - NSDictionary / Python dict（键为 X/Y/Width/Height 或 x/y/width/height）
    本函数统一处理两种形式，避免 'dict has no attribute origin' 崩溃。
    """
    if b is None:
        return None
    # 形式 1：CGRect/NSRect
    origin = getattr(b, "origin", None)
    size = getattr(b, "size", None)
    if origin is not None and size is not None:
        return (float(origin.x), float(origin.y),
                float(size.width), float(size.height))
    # 形式 2：字典（Apple 文档标准键名为 X/Y/Width/Height）
    if isinstance(b, dict):
        def _get(*keys):
            for k in keys:
                if k in b:
                    try:
                        return float(b[k])
                    except Exception:
                        return None
            return None
        x = _get("X", "x")
        y = _get("Y", "y")
        w = _get("Width", "width")
        h = _get("Height", "height")
        if x is not None and y is not None and w is not None and h is not None:
            return (x, y, w, h)
    return None


@dataclass
class Display:
    index: int
    raw_left: int
    raw_top: int
    raw_width: int
    raw_height: int
    pt_x: float
    pt_y: float
    pt_width: float
    pt_height: float
    scale: float


@dataclass
class RunningApp:
    pid: int
    name: str
    bundle: str
    has_window: bool


class CaptureManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.displays: List[Display] = []
        self._mss = None
        # 每个线程独立的 mss 实例：macOS 上 CoreGraphics 抓取不能跨线程
        # 复用主线程创建的实例，否则后台工作线程抓到的帧会是黑屏。
        self._local = threading.local()
        if _MSS_OK:
            try:
                self._mss = mss.MSS()
                self._build_displays()
            except Exception as e:  # pragma: no cover
                print("mss 初始化失败:", e)
                self._mss = None

    def _thread_mss(self):
        """返回当前线程自己的 mss 实例（懒创建并缓存于线程局部存储）。"""
        if not _MSS_OK:
            return None
        inst = getattr(self._local, "mss", None)
        if inst is None:
            inst = mss.MSS()
            self._local.mss = inst
        return inst

    # ---- 显示器映射 ----
    def _build_displays(self):
        self.displays = []
        monitors = self._mss.monitors[1:] if self._mss and len(self._mss.monitors) > 1 else []
        quartz_displays = self._quartz_displays() if _QUARTZ_OK else []
        # 用"原始宽高"配对 mss monitor 与 Quartz display
        for i, mon in enumerate(monitors):
            rw, rh = mon["width"], mon["height"]
            qd = next((q for q in quartz_displays if q[1] == rw and q[2] == rh), None)
            if qd:
                bx, by, bw, bh, scale = qd[3], qd[4], qd[5], qd[6], qd[7]
            else:
                # 兜底：假设 scale=1
                scale = 1.0
                bx, by, bw, bh = mon["left"], mon["top"], rw, rh
            self.displays.append(Display(
                index=i, raw_left=mon["left"], raw_top=mon["top"],
                raw_width=rw, raw_height=rh,
                pt_x=bx, pt_y=by, pt_width=bw, pt_height=bh, scale=scale))

    @staticmethod
    def _quartz_displays():
        out = []
        try:
            (_, ids, _) = CGGetOnlineDisplayList(32, None, None)
            for did in ids:
                b = CGDisplayBounds(did)
                pw = CGDisplayPixelsWide(did)
                ph = CGDisplayPixelsHigh(did)
                bw = b.size.width
                bh = b.size.height
                scale = (pw / bw) if bw else 1.0
                out.append((did, pw, ph, b.origin.x, b.origin.y, bw, bh, scale))
        except Exception as e:  # pragma: no cover
            print("读取显示器失败:", e)
        return out

    # ---- 坐标转换：全局点坐标 -> mss 原始像素 ----
    def points_to_raw(self, x: float, y: float, w: float, h: float):
        disp = self._display_for_point(x + w / 2, y + h / 2)
        if disp is None:
            # 兜底：用主显示器 scale
            disp = self.displays[0] if self.displays else None
            if disp is None:
                return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        rx = disp.raw_left + (x - disp.pt_x) * disp.scale
        ry = disp.raw_top + (y - disp.pt_y) * disp.scale
        return {
            "left": int(round(rx)),
            "top": int(round(ry)),
            "width": int(round(w * disp.scale)),
            "height": int(round(h * disp.scale)),
        }

    def _display_for_point(self, px: float, py: float) -> Optional[Display]:
        for d in self.displays:
            if (d.pt_x <= px < d.pt_x + d.pt_width and
                    d.pt_y <= py < d.pt_y + d.pt_height):
                return d
        return None

    # ---- 抓取 ----
    def capture_raw(self, raw_rect: dict) -> Optional[np.ndarray]:
        """按 mss 原始像素区域抓取，返回 BGR numpy 图。加锁保证线程安全。"""
        m = self._thread_mss()
        if m is None:
            return None
        try:
            with self._lock:
                shot = m.grab(raw_rect)
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
                shot.height, shot.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:  # pragma: no cover
            print("抓取失败:", e)
            return None

    def capture_points(self, x, y, w, h) -> Optional[np.ndarray]:
        return self.capture_raw(self.points_to_raw(x, y, w, h))

    # ---- 运行应用枚举 ----
    def list_running_apps(self) -> List[RunningApp]:
        apps: List[RunningApp] = []
        if not _QUARTZ_OK:
            return apps
        try:
            for a in NSWorkspace.sharedWorkspace().runningApplications():
                if a.activationPolicy() != 0:  # 仅普通前台应用
                    continue
                pid = int(a.processIdentifier())
                name = str(a.localizedName() or "")
                bundle = str(a.bundleIdentifier() or "")
                has_window = self._app_has_window(pid)
                apps.append(RunningApp(pid=pid, name=name, bundle=bundle, has_window=has_window))
        except Exception as e:  # pragma: no cover
            print("枚举应用失败:", e)
        return apps

    @staticmethod
    def _app_has_window(pid: int) -> bool:
        try:
            wins = CGWindowListCopyWindowInfo(
                (1 << 0), 0)  # kCGWindowListOptionOnScreenOnly
            for w in wins:
                if int(w.get(kCGWindowOwnerPID, -1)) == pid:
                    layer = w.get(kCGWindowLayer, 0)
                    rect = _bounds_to_rect(w.get(kCGWindowBounds))
                    if layer == 0 and rect and rect[2] > 50 and rect[3] > 50:
                        return True
        except Exception:
            return False
        return False

    def get_app_main_window_rect(self, pid: int, name_hint: str = None):
        """返回该应用主窗口的全局"点坐标"矩形 (x,y,w,h)。

        多进程应用（如 Tauri / Electron / WebView 类）的可见窗口可能被 helper 进程拥有，
        主进程 pid 搜不到。这里按 tier 回退：
          1. 先按 pid 精确匹配；同时枚举同 bundle 的所有进程（连带 helpers）。
          2. 若失败，按 kCGWindowOwnerName 匹配 name_hint。
          3. 若仍失败，全部可见窗口中按层最低 + 面积最大挑一个候选。
        返回最合适的一个 (x,y,w,h)，找不到返回 None。
        """
        if not _QUARTZ_OK:
            return None
        try:
            wins = CGWindowListCopyWindowInfo((1 << 0), 0)
        except Exception as e:  # pragma: no cover
            print("读取窗口失败:", e)
            return None
        # -2 哨兵：尝试抓取"当前最前台应用"的窗口（作为兜底）
        if pid == -2:
            try:
                fa = NSWorkspace.sharedWorkspace().frontmostApplication()
                if fa:
                    pid = int(fa.processIdentifier())
                    if not name_hint:
                        name_hint = str(fa.localizedName() or "")
            except Exception:
                pass
        # ---- 同包相关的所有进程（应用 + helpers）----
        sibling_pids = self._sibling_pids_for(pid, name_hint)
        # Tier 1: pid / sibling pids
        cand = self._best_window(wins, match_pids=sibling_pids)
        if cand:
            return cand
        # Tier 2: by name
        if name_hint:
            cand = self._best_window(wins, match_name=name_hint)
            if cand:
                return cand
        # Tier 3: 任何 layer<=100 且面积足够大的
        return self._best_window(wins, match_pids=None, max_layer=100)

    @staticmethod
    def _best_window(wins, match_pids=None, match_name=None, max_layer=300):
        """在 wins 里挑最优窗口：(层最低，面积最大)。"""
        best = None
        best_score = None  # (layer, -area)
        name_low = (match_name or "").lower()
        for w in wins:
            try:
                layer = int(w.get(kCGWindowLayer, 9999))
            except Exception:
                layer = 9999
            if layer > max_layer:
                continue
            if match_pids is not None:
                try:
                    op = int(w.get(kCGWindowOwnerPID, -1))
                except Exception:
                    op = -1
                if op not in match_pids:
                    continue
            elif name_low:
                owner = str(w.get("kCGWindowOwnerName", "") or "")
                title = str(w.get("kCGWindowName", "") or "")
                if name_low not in owner.lower() and name_low not in title.lower():
                    continue
            rect = _bounds_to_rect(w.get(kCGWindowBounds))
            if not rect:
                continue
            x, y, ww, hh = rect
            if ww < 50 or hh < 50:
                continue
            score = (layer, -ww * hh)
            if best_score is None or score < best_score:
                best_score = score
                best = (int(x), int(y), int(ww), int(hh))
        return best

    def _sibling_pids_for(self, pid, name_hint=None):
        """返回与给定 pid 同 bundle 的所有 pid（含主进程 + helpers）。"""
        pids = {int(pid)}
        if not _QUARTZ_OK:
            return pids
        try:
            bundle = None
            for a in NSWorkspace.sharedWorkspace().runningApplications():
                if int(a.processIdentifier()) == int(pid):
                    bundle = a.bundleIdentifier()
                    break
            if bundle:
                for a in NSWorkspace.sharedWorkspace().runningApplications():
                    if str(a.bundleIdentifier() or "") == bundle:
                        pids.add(int(a.processIdentifier()))
        except Exception:
            pass
        return pids

    @property
    def ready(self) -> bool:
        return self._mss is not None

    @property
    def quartz_available(self) -> bool:
        return _QUARTZ_OK


# 单例
_manager = None


def get_manager() -> CaptureManager:
    global _manager
    if _manager is None:
        _manager = CaptureManager()
    return _manager


# ---------------------------------------------------------------------------
# 屏幕录制（TCC）权限
# ---------------------------------------------------------------------------
# 根因说明：macOS 上未授予『屏幕录制』权限时，截图里其它 App 的窗口会被系统
# 涂成纯色/空白。于是监控区域一旦被别的窗口遮挡，那块画面就"消失"，模板永远
# 匹配不上 —— 表现为『有窗口遮挡之后检测失效』。授权后即可抓到完整合成画面，
# 遮挡不再导致检测失效（注意：目标被完全遮住时仍无法检测，属物理限制）。
def screen_capture_access_status() -> str:
    """返回 'granted' / 'denied' / 'unknown'。

    'unknown' 表示非 macOS 或旧系统（无该 API），此时假定权限可用。
    """
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return "granted" if bool(CGPreflightScreenCaptureAccess()) else "denied"
    except Exception:
        return "unknown"


def request_screen_capture_access() -> None:
    """主动触发系统『屏幕录制』授权弹窗（已授权时为 no-op，不报错）。

    应在 App 处于前台时调用一次，以便首次使用时弹出系统授权框。
    """
    try:
        from Quartz import CGRequestScreenCaptureAccess
        CGRequestScreenCaptureAccess()
    except Exception as e:  # pragma: no cover
        print("申请屏幕录制权限失败:", e)


if __name__ == "__main__":
    m = get_manager()
    print("mss ready:", m.ready, "| quartz:", m.quartz_available)
    print("displays:", [(d.index, d.raw_width, d.raw_height, round(d.scale, 2)) for d in m.displays])
    if m.quartz_available:
        apps = m.list_running_apps()
        print("运行应用数:", len(apps))
        for a in apps[:5]:
            print("  -", a.pid, a.name, "window:", a.has_window)
