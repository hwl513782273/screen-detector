# -*- coding: utf-8 -*-
"""
框选屏幕检测工具 —— 核心检测与自学习引擎（无界面，可 headless 测试）

职责：
  - 多模板匹配（OpenCV matchTemplate，支持多尺度）
  - 检测状态机：出现 -> 响铃；消失 -> 自动停铃；手动停止 -> 停铃
  - 基于用户反馈的自学习：动态调整每模板匹配阈值 + 模板精炼，降低误报、提升命中率

设计要点：
  - 匹配在灰度图上进行，更快更稳。
  - 每个模板独立维护：命中得分列表 / 误报得分列表 / 精炼模板。
  - 阈值自学习：保证"已确认为误报"的样本得分一定不过线，同时尽量靠近正样本分布。
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

# 多尺度搜索：覆盖 0.5x~1.5x，使"载入图片"与原图分辨率/渲染略有差异时也能命中
DEFAULT_SCALES = (0.5, 0.6, 0.75, 0.85, 0.92, 1.0, 1.08, 1.15, 1.3, 1.5)


@dataclass
class Template:
    """一个待检测图案模板。"""
    tid: str
    name: str
    image: np.ndarray               # BGR 彩色图（原始/基础模板）
    enabled: bool = True
    base_threshold: float = 0.75    # 用户设定的基准阈值（默认更灵敏，配合载入图自动校准）
    pos_scores: List[float] = field(default_factory=list)   # 已确认命中的匹配得分
    neg_scores: List[float] = field(default_factory=list)   # 已确认误报的匹配得分
    refined_image: Optional[np.ndarray] = None  # 精炼后的模板（彩色）
    use_refined: bool = True        # 是否使用精炼模板参与匹配
    sound: str = ""                 # 该图案命中时使用的提示音（空字符串=用全局默认声音）
    _scale_cache: dict = field(default_factory=dict, repr=False)  # 多尺度预计算缓存

    # ---- 派生属性 ----
    @property
    def active_image(self) -> np.ndarray:
        """实际用于匹配的模板（精炼优先）。"""
        if self.use_refined and self.refined_image is not None:
            return self.refined_image
        return self.image

    @property
    def effective_threshold(self) -> float:
        """自学习后的有效阈值。"""
        return compute_threshold(self.base_threshold, self.pos_scores, self.neg_scores)

    def gray(self) -> np.ndarray:
        return to_gray(self.active_image)

    def reset_learning(self) -> None:
        self.pos_scores = []
        self.neg_scores = []
        self.refined_image = None
        self._scale_cache.clear()


@dataclass
class MatchResult:
    tid: str
    name: str
    score: float
    x: int
    y: int
    w: int
    h: int
    crop: np.ndarray               # 命中处裁切（彩色），用于反馈展示/模板精炼


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def match_score(fg: np.ndarray, tg: np.ndarray,
                use_abs: bool = True) -> Tuple[float, Tuple[int, int]]:
    """
    在 fg 中匹配模板 tg，返回 (得分, 最佳位置)。
    得分统一为"越高越像"(0~1)。
    对近纯色(零方差)模板，TM_CCOEFF_NORMED 会因除以标准差而失效，
    自动回退到 TM_SQDIFF_NORMED 并取 1-误差 作为得分。

    use_abs=True 时取相关系数的绝对值，对"深色底白字"与"白色底黑字"
    这类反色 UI 状态更鲁棒。
    """
    tv = float(np.std(tg))
    if tv < 1.0:  # 近乎纯色模板
        # 注意 cv2.minMaxLoc 返回 (minVal, maxVal, minLoc, maxLoc)
        res = cv2.matchTemplate(fg, tg, cv2.TM_SQDIFF_NORMED)
        minv, _, minloc, _ = cv2.minMaxLoc(res)
        return (1.0 - float(minv)), (int(minloc[0]), int(minloc[1]))
    res = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    score = float(maxv)
    if use_abs:
        score = abs(score)
    return score, (int(maxloc[0]), int(maxloc[1])) 


def compute_threshold(base: float, pos: List[float], neg: List[float]) -> float:
    """
    根据反馈样本计算有效阈值：
      - 若有误报样本：阈值至少 = 最高误报得分 + 余量，保证已确认的误报不再触发。
      - 若有命中样本：阈值不高于 最低命中得分 - 余量，保证已确认的命中仍触发。
      - 两者皆有时取区间中点（偏向更高以减少误报）。
    最终与基准阈值取较严格者并夹在 [0.50, 0.99]。
    """
    margin = 0.01
    lo = 0.50
    hi = 0.99
    if neg and pos:
        t = (max(neg) + min(pos)) / 2.0
        # 保证误报不过线、命中仍过线
        t = max(t, max(neg) + margin)
        t = min(t, min(pos) - margin)
    elif neg:
        t = max(neg) + margin
    elif pos:
        t = min(pos) - margin
    else:
        t = base
    # 与用户基准取较严格者（用户调高基准应被尊重）
    t = max(t, base) if base > t else t
    return float(min(hi, max(lo, t)))


# ---------------------------------------------------------------------------
# 检测引擎
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self, scales: Tuple[float, ...] = DEFAULT_SCALES):
        self.scales = tuple(scales)
        self.templates: List[Template] = []
        self._lock = threading.Lock()
        self._scale_sig: Tuple[float, ...] = ()

    # ---- 模板管理 ----
    def add_template(self, name: str, image: np.ndarray) -> Template:
        tid = f"t{len(self.templates)+1}_{int(np.random.rand()*1e6)}"
        t = Template(tid=tid, name=name, image=image.astype(np.uint8))
        with self._lock:
            self.templates.append(t)
        return t

    def remove_template(self, tid: str) -> None:
        with self._lock:
            self.templates = [t for t in self.templates if t.tid != tid]

    def get(self, tid: str) -> Optional[Template]:
        with self._lock:
            return next((t for t in self.templates if t.tid == tid), None)

    def set_enabled(self, tid: str, enabled: bool) -> None:
        t = self.get(tid)
        if t:
            t.enabled = enabled

    def rename(self, tid: str, name: str) -> None:
        t = self.get(tid)
        if t:
            t.name = name

    def reset_learning(self, tid: str) -> None:
        t = self.get(tid)
        if t:
            t.reset_learning()

    def set_scales(self, scales: Tuple[float, ...]) -> None:
        with self._lock:
            self.scales = tuple(scales)
            for t in self.templates:
                t._scale_cache.clear()

    # ---- 多尺度预计算 ----
    def _prepared(self, t: Template) -> List[Tuple[float, np.ndarray]]:
        sig = self.scales
        if t._scale_cache.get("sig") != sig:
            # 关键：模板灰度必须归一为 uint8。反馈精炼后 refined_image 是
            # float32，若直接拿去 matchTemplate 会与 uint8 的帧灰度类型不一致，
            # 触发 OpenCV 断言崩溃 —— 这会让后台工作线程静默死亡（表现为
            # "画面里明明有目标、测试匹配能命中，但实时监控却从不提示"）。
            g = to_gray(t.active_image)
            if g.dtype != np.uint8:
                g = np.clip(g, 0, 255).astype(np.uint8)
            gray = g
            prepared = []
            h, w = gray.shape[:2]
            for s in sig:
                nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
                if nw >= w and nh >= h or (nw <= w and nh <= h):
                    rs = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                    prepared.append((s, rs))
            t._scale_cache = {"sig": sig, "data": prepared}
        return t._scale_cache["data"]

    # ---- 单帧匹配 ----
    def match_frame(self, frame_bgr: np.ndarray,
                    only_tids: Optional[List[str]] = None,
                    return_debug: bool = False):
        """
        对一帧画面做全部（或指定）启用模板的多尺度匹配。
        默认返回所有过线结果；return_debug=True 时额外返回每个模板的
        最佳得分/位置/尺度，便于 UI 诊断"未命中"原因。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            if return_debug:
                return [], {}
            return []
        fg = to_gray(frame_bgr)
        fh, fw = fg.shape[:2]
        results: List[MatchResult] = []
        debug: dict = {}
        # 全程持锁：防止与 record_feedback/_refine/set_scales 并发读写
        # 模板的 active_image / _scale_cache，避免在后台工作线程中抛出
        # KeyError 或读到半更新的精炼模板，导致工作线程静默崩溃（表现为
        # "明明画面里有目标却从不提示命中/不响铃"）。
        with self._lock:
            templates = list(self.templates)
            for t in templates:
                if not t.enabled:
                    continue
                if only_tids is not None and t.tid not in only_tids:
                    continue
                best_score = -1.0
                best_loc = (0, 0)
                best_scale = 1.0
                for s, tg in self._prepared(t):
                    th, tw = tg.shape[:2]
                    if fh < th or fw < tw:
                        continue
                    sc, loc = match_score(fg, tg)
                    if sc > best_score:
                        best_score = sc
                        best_loc = loc
                        best_scale = s
                thresh = t.effective_threshold
                debug[t.tid] = {
                    "name": t.name,
                    "best_score": float(best_score),
                    "threshold": float(thresh),
                    "loc": best_loc,
                    "scale": float(best_scale),
                }
                if best_score >= thresh:
                    tw = int(round(t.active_image.shape[1] * best_scale))
                    th = int(round(t.active_image.shape[0] * best_scale))
                    x, y = best_loc
                    x2, y2 = min(x + tw, fw), min(y + th, fh)
                    crop = frame_bgr[y:y2, x:x2].copy()
                    results.append(MatchResult(
                        tid=t.tid, name=t.name, score=float(best_score),
                        x=int(x), y=int(y), w=int(x2 - x), h=int(y2 - y), crop=crop))
        # 按得分降序，方便 UI 展示
        results.sort(key=lambda r: r.score, reverse=True)
        if return_debug:
            return results, debug
        return results

    def debug_match_scores(self, frame_bgr: np.ndarray,
                           only_tids: Optional[List[str]] = None) -> dict:
        """仅返回每个模板的最佳匹配调试信息（不过滤阈值）。"""
        _, debug = self.match_frame(frame_bgr, only_tids=only_tids, return_debug=True)
        return debug

    # ---- 反馈学习 ----
    def record_feedback(self, tid: str, is_hit: bool, score: float,
                        crop: Optional[np.ndarray] = None) -> dict:
        """
        记录一次反馈并学习：
          - 命中：加入 pos_scores；并用裁切精炼模板（与现有精炼/基础模板对齐平均）。
          - 误报：加入 neg_scores。
        返回更新后的学习摘要。
        """
        t = self.get(tid)
        if t is None:
            return {}
        with self._lock:
            if is_hit:
                t.pos_scores.append(float(score))
                if crop is not None and crop.size:
                    self._refine(t, crop)
            else:
                t.neg_scores.append(float(score))
        return {
            "tid": tid,
            "effective_threshold": t.effective_threshold,
            "pos_count": len(t.pos_scores),
            "neg_count": len(t.neg_scores),
            "refined": t.refined_image is not None,
        }

    def _refine(self, t: Template, crop: np.ndarray) -> None:
        """用确认命中的裁切对模板做加权平均精炼（对齐到模板尺寸）。"""
        th, tw = t.active_image.shape[:2]
        crop_r = cv2.resize(crop.astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)
        if t.refined_image is None:
            t.refined_image = crop_r.astype(np.float32)
        else:
            alpha = 0.15  # 缓慢靠拢，避免单次噪声主导
            t.refined_image = (t.refined_image * (1 - alpha) + crop_r.astype(np.float32) * alpha)
        t._scale_cache.clear()  # 模板变了，刷新多尺度缓存

    def summary(self) -> List[dict]:
        with self._lock:
            return [{
                "tid": t.tid, "name": t.name, "enabled": t.enabled,
                "base_threshold": t.base_threshold,
                "effective_threshold": t.effective_threshold,
                "pos": len(t.pos_scores), "neg": len(t.neg_scores),
                "refined": t.refined_image is not None,
            } for t in self.templates]


if __name__ == "__main__":
    # 简单自测：合成带纹理图案 + 噪声背景，验证匹配与阈值自学习
    np.random.seed(0)

    def make_pattern():
        img = np.zeros((60, 60, 3), np.uint8)
        cv2.rectangle(img, (5, 5), (55, 55), (0, 180, 220), -1)   # 青色底
        cv2.rectangle(img, (20, 20), (40, 40), (230, 30, 30), -1) # 红色内块（带纹理/方差）
        cv2.putText(img, "X", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    pattern = make_pattern()
    d = Detector()
    tpl = d.add_template("图案", pattern.copy())
    # 把图案放到画面中 -> 应命中
    scene = np.full((400, 400, 3), 40, np.uint8)
    scene[200:260, 200:260] = pattern
    res = d.match_frame(scene)
    print("命中测试:", [(r.name, round(r.score, 3)) for r in res])
    assert res and res[0].tid == tpl.tid, "应检测到图案"
    # 纯噪声 -> 不应命中（阈值默认 0.85）
    noise = np.random.randint(0, 255, (400, 400, 3), np.uint8)
    res2 = d.match_frame(noise)
    print("噪声测试:", [(r.name, round(r.score, 3)) for r in res2])
    assert not res2, "噪声不应误报"
    # 学习能力：把一次"边界误报"(得分 0.88)记为误报 -> 阈值应提高并高于该得分
    d.record_feedback(tpl.tid, is_hit=False, score=0.88)
    print("学习后有效阈值:", round(tpl.effective_threshold, 3))
    assert tpl.effective_threshold > 0.88, "误报样本得分应不过线(阈值需高于它)"
    # 命中反馈应精炼模板
    d.record_feedback(tpl.tid, is_hit=True, score=res[0].score, crop=res[0].crop)
    assert tpl.refined_image is not None, "命中反馈后应生成精炼模板"
    # compute_threshold 边界：正负样本共存时取区间中点并夹在合理范围
    from detector_core import compute_threshold
    assert 0.88 < compute_threshold(0.85, [0.95], [0.88]) < 0.95
    print("学习摘要:", d.summary())
    print("OK: 核心引擎自测通过")
