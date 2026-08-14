# -*- coding: utf-8 -*-
"""
框选屏幕检测工具 —— 主入口

运行方式：
  1) 直接运行（开发/调试）：  python screen_detector.py
  2) 打包后用 .app 启动（最终交付）
"""
from __future__ import annotations

import sys
import os

# 让 PyInstaller 打包后也能找到同目录模块
if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))

from ui import main

if __name__ == "__main__":
    main()
