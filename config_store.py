# -*- coding: utf-8 -*-
"""
配置与模板持久化（JSON + PNG）。
模板图像、学习反馈、参数都会落盘，跨会话累积学习成果。
"""
from __future__ import annotations

import json
import os
import shutil

import cv2
import numpy as np

APP_SUPPORT = os.path.expanduser("~/Library/Application Support/ScreenPatternDetector")
CONFIG_PATH = os.path.join(APP_SUPPORT, "config.json")


class ConfigStore:
    def __init__(self, base: str = APP_SUPPORT):
        self.base = base
        os.makedirs(self.base, exist_ok=True)
        self.data = {
            "region": None,            # [x, y, w, h] 点坐标
            "interval_ms": 500,
            "cooldown_s": 3,
            "multiscale": True,
            "sound": "Ping",
            "templates": [],           # 见 _serialize_template
        }

    # ---- 模板图像存取 ----
    def _img_path(self, tid: str) -> str:
        return os.path.join(self.base, f"{tid}.png")

    def save_image(self, tid: str, img: np.ndarray) -> None:
        cv2.imwrite(self._img_path(tid), img)

    def load_image(self, tid: str) -> Optional[np.ndarray]:
        p = self._img_path(tid)
        if os.path.exists(p):
            return cv2.imread(p)
        return None

    def delete_image(self, tid: str) -> None:
        p = self._img_path(tid)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # ---- 序列化 ----
    def _serialize_template(self, t) -> dict:
        # 仅保存基础模板图像；精炼模板也保存以便复用
        self.save_image(t.tid, t.image)
        if t.refined_image is not None:
            self.save_image(t.tid + "_refined", t.refined_image.astype(np.uint8))
        return {
            "tid": t.tid, "name": t.name, "enabled": t.enabled,
            "base_threshold": t.base_threshold,
            "pos_scores": t.pos_scores, "neg_scores": t.neg_scores,
            "use_refined": t.use_refined,
            "has_refined": t.refined_image is not None,
            "sound": getattr(t, "sound", "") or "",
        }

    def save(self, detector, region, params: dict) -> None:
        self.data["region"] = region
        self.data.update(params)
        self.data["templates"] = [self._serialize_template(t) for t in detector.templates]
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        shutil.move(tmp, CONFIG_PATH)

    def load(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass
        return self.data

    # ---- 临时文件 / 孤立文件自动清理（v6.24）----
    def gc_orphans(self) -> int:
        """扫描配置目录，清理『不再被模板列表引用』的残留文件，防止长期占用存储。

        清理对象：
          1. 孤立的模板图片：<tid>.png / <tid>_refined.png —— 模板已从列表删除，
             但图片文件残留（删除模板时正常会删，此处兜底清理异常遗留）。
          2. 上次异常中断遗留的 *.tmp 文件（save() 用 .tmp 原子替换，正常不应残留）。
        返回清理掉的文件数量（仅用于日志）。不会删除当前模板引用的任何文件，
        也不会删除 config.json 本身。

        ⚠️ 安全保护：当存档模板数组为空（配置损坏 / 初次使用 / 加载失败）时，
        绝不删除任何模板图片——否则会把用户真实模板当成孤儿一次性清空。
        仅在『模板数组非空』时才清理确实不再被引用的图片；.tmp 始终清理。
        """
        removed = 0
        try:
            templates = self.data.get("templates", []) or []
            tids = {str(t.get("tid")) for t in templates if t.get("tid")}
            safe_to_clean_png = len(tids) > 0   # 模板数组为空时，保护所有图片不被误删
            for fn in os.listdir(self.base):
                full = os.path.join(self.base, fn)
                if not os.path.isfile(full):
                    continue
                if fn.endswith(".tmp"):
                    try:
                        os.remove(full)
                        removed += 1
                    except OSError:
                        pass
                    continue
                if fn.endswith(".png") and safe_to_clean_png:
                    stem = fn[:-4]
                    if stem.endswith("_refined"):
                        stem = stem[:-8]
                    if stem and stem not in tids:
                        try:
                            os.remove(full)
                            removed += 1
                        except OSError:
                            pass
        except Exception as e:  # pragma: no cover
            print("清理孤立/临时文件失败:", e)
        return removed

    def restore_templates(self, detector) -> None:
        """把存档模板（含图像与学习数据）恢复到 detector。"""
        for td in self.data.get("templates", []):
            base_img = self.load_image(td["tid"])
            if base_img is None:
                continue
            t = detector.add_template(td["name"], base_img)
            t.tid = td["tid"]
            t.enabled = td.get("enabled", True)
            t.base_threshold = td.get("base_threshold", 0.85)
            t.pos_scores = td.get("pos_scores", [])
            t.neg_scores = td.get("neg_scores", [])
            t.use_refined = td.get("use_refined", True)
            t.sound = td.get("sound", "") or ""
            if td.get("has_refined"):
                ref = self.load_image(td["tid"] + "_refined")
                if ref is not None:
                    t.refined_image = ref.astype(np.float32)
            t._scale_cache.clear()
