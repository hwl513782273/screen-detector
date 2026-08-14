# -*- coding: utf-8 -*-
"""macOS 原生文字识别（Apple Vision OCR）。

- 优先使用系统自带的 Vision 框架（pyobjc-framework-Vision），无需外部二进制、中文支持好。
- 不可用时（未安装该框架）降级为不可用状态，由 UI 给出清晰提示。

注意 CGDataProvider 的存活问题：
  - CGDataProviderCreateWithBytesNoCopy 引用传入的 bytes，函数返回后该内存可能被回收，
    导致 OCR 读到垃圾数据（use-after-free）。该符号在本机 Quartz 中也不存在。
  - 改用 CGDataProviderCreateWithCFData：内部会拷贝数据，生命周期安全，且不依赖符号存在性。
"""
from __future__ import annotations

import cv2
import numpy as np
import re
import unicodedata

try:
    from Vision import VNRecognizeTextRequest, VNImageRequestHandler
    from Quartz import (
        CGImageCreate, CGColorSpaceCreateDeviceRGB,
        kCGImageAlphaNoneSkipLast, kCGBitmapByteOrderDefault,
        CGDataProviderCreateWithCFData,
    )
    from CoreFoundation import CFDataCreate
    _VISION_OK = True
except Exception:  # pragma: no cover
    _VISION_OK = False

try:
    # 显式自动释放池：每次 OCR 创建的 CGImage / VNImageRequestHandler /
    # VNRecognizeTextRequest 等 Obj-C 对象若不 drain，长时间运行会持续累积，
    # 最终被 macOS 内存压力杀掉（表现为『运行一段时间之后闪退』）。
    from Foundation import NSAutoreleasePool
    _HAVE_AUTORELEASE_POOL = True
except Exception:  # pragma: no cover
    _HAVE_AUTORELEASE_POOL = False


def vision_available() -> bool:
    return _VISION_OK


def _normalize_text(text: str) -> str:
    """文字归一化：统一全角/半角、移除空格与零宽字符，降低 OCR 分词抖动导致漏配。

    Apple Vision 对中文有时会在字符间插入空格（如把"刚刚"识别成"刚 刚"），
    直接子串匹配会漏配。归一化后再匹配可消除这类抖动。
    """
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    # 移除所有空白类字符（含全角空格 U+3000）与零宽字符
    text = re.sub(r"[\s\u200b-\u200f\ufeff\u3000]+", "", text)
    return text.strip()


def _frame_to_cgimage(frame_bgr):
    """把 BGR numpy 图转成 CGImage（Vision 需要）。返回 cgimage 对象。

    全程使用 CFData 拷贝，避免底层 buffer 被回收导致的乱码。
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_rgba = np.empty((h, w, 4), dtype=np.uint8)
    img_rgba[..., 0:3] = img_rgb
    img_rgba[..., 3] = 255
    data = img_rgba.tobytes()
    # CFDataCreate 会拷一份数据，provider 持有副本，无需外部保持引用
    cfdata = CFDataCreate(None, data, len(data))
    provider = CGDataProviderCreateWithCFData(cfdata)
    colorspace = CGColorSpaceCreateDeviceRGB()
    cg = CGImageCreate(
        w, h, 8, 32, w * 4, colorspace,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault,
        provider, None, False, 0,
    )
    return cg


def _text_of(obs):
    """从 VNRecognizedTextObservation 取识别文字，兼容不同 macOS 版本 API。"""
    try:
        # 新系统支持 .text() 便捷属性
        return str(obs.text())
    except Exception:
        pass
    try:
        cands = obs.topCandidates_(1)
        if cands:
            return str(cands[0].string())
    except Exception:
        pass
    return ""


def recognize_text(frame_bgr, max_edge=1280):
    """识别画面中的文字，返回 [{"text","x","y","w","h"}]（坐标基于传入帧『原图』像素）。

    boundingBox 为归一化坐标，origin 在左下角，需要翻转 y。

    CPU/内存优化（v6.24）：Apple Vision OCR 的计算量大致随像素面积增长。多数屏幕
    文字在较长边 ≤ 1280px 时识别精度几乎不变，因此这里先把帧按最长边等比缩小到
    max_edge 以内再送 OCR，识别出的归一化坐标乘以缩放比映射回『原图』坐标系——对外
    坐标空间不变（调用方仍在原图上绘制），但内部 RGBA 缓冲 / CFData / Vision 计算量
    随面积下降（最长边减半时计算量约降 75%）。min(max_edge, 原最长边) 保证不放大。

    注意：函数体包在显式 NSAutoreleasePool 内——每次 OCR 都会创建 CGImage /
    VNImageRequestHandler / VNRecognizeTextRequest 等 Obj-C 对象，若不及时 drain，
    长时间持续识别会让这些临时对象无限累积，最终被 macOS 内存压力杀掉（闪退）。
    显式 drain 后，Python 侧仅保留已转成 str/int 的结果，无 Obj-C 引用残留。
    """
    if not _VISION_OK or frame_bgr is None:
        return []
    hh, ww = frame_bgr.shape[:2]
    scale = 1.0
    if max_edge and max(hh, ww) > max_edge:
        scale = float(max_edge) / float(max(hh, ww))
    small = frame_bgr
    if scale < 1.0:
        # 缩小识别图：面积减小 -> Vision 处理更快、内存更省，文字检测精度基本不受影响
        small = cv2.resize(frame_bgr,
                           (int(round(ww * scale)), int(round(hh * scale))),
                           interpolation=cv2.INTER_AREA)
    pool = NSAutoreleasePool.alloc().init() if _HAVE_AUTORELEASE_POOL else None
    try:
        h, w = small.shape[:2]
        cg = _frame_to_cgimage(small)
        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        req = VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])
        req.setRecognitionLevel_(0)  # 0=精确，1=快速
        req.setUsesLanguageCorrection_(True)
        handler.performRequests_error_([req], None)
        obs = req.results()
        out = []
        for o in obs or []:
            txt = _text_of(o)
            if not txt:
                continue
            b = o.boundingBox()  # 归一化，origin 在左下角
            x = int(b.origin.x * w)
            y = int((1.0 - b.origin.y - b.size.height) * h)
            bw = int(b.size.width * w)
            bh = int(b.size.height * h)
            # 把『缩小图』坐标映射回『原图』坐标（对外坐标空间不变）
            if scale < 1.0:
                x = int(x / scale); y = int(y / scale)
                bw = int(bw / scale); bh = int(bh / scale)
            out.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})
        return out
    except Exception as e:  # pragma: no cover
        print("OCR 失败:", e)
        return []
    finally:
        # 释放本轮 OCR 产生的所有 Obj-C 临时对象，避免内存累积导致闪退
        if pool is not None:
            del pool


def text_matches(found, targets, mode):
    """判断识别到的文字是否命中目标。

    found:   识别到的文字字符串列表
    targets: 目标文字列表
    mode:    'any' 包含任一 / 'all' 全部包含 / 'equal' 完全相等(任一目标整串出现)
    返回 (是否命中, 命中文字列表)。

    注意：匹配前会对文字做归一化（移除空格、零宽字符、全角转半角），
    因此命中列表返回的是归一化后的目标字符串，便于 UI 展示。
    """
    if not targets:
        return False, []
    found = [_normalize_text(t) for t in found if _normalize_text(t)]
    tgt = [_normalize_text(t) for t in targets if _normalize_text(t)]
    if not tgt:
        return False, []
    if mode == "equal":
        matched = [t for t in tgt if t in found]
        return len(matched) > 0, matched
    if mode == "all":
        matched = [t for t in tgt if any(t in f for f in found)]
        return len(matched) == len(tgt), matched
    # any
    matched = [t for t in tgt if any(t in f for f in found)]
    return len(matched) > 0, matched
