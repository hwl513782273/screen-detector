# -*- mode: python ; coding: utf-8 -*-
# 框选屏幕检测工具 打包配置（macOS .app）
import os
# 兼容构建开关（不影响默认 v6.49 的 arm64 构建）：
#   COMPAT_ARCH      : EXE 的目标架构，默认 None（跟随当前机器=arm64）；兼容包设 'x86_64'
#   COMPAT_VERSION   : CFBundleShortVersionString，默认 '6.49'
#   COMPAT_MIN_SYS   : LSMinimumSystemVersion，默认 '10.15'
COMPAT_ARCH = os.environ.get("COMPAT_ARCH", None)
COMPAT_VERSION = os.environ.get("COMPAT_VERSION", "6.49")
COMPAT_MIN_SYS = os.environ.get("COMPAT_MIN_SYS", "10.15")
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    "Quartz", "Cocoa", "AppKit", "objc",
    "mss", "cv2", "numpy", "PIL",
]
# pyobjc 采用惰性子模块加载，需显式收集以避免运行期 ImportError
hiddenimports += collect_submodules("Quartz")
hiddenimports += collect_submodules("Cocoa")
hiddenimports += collect_submodules("objc")
hiddenimports += collect_submodules("Vision")
hiddenimports += collect_submodules("Foundation")

a = Analysis(
    ["screen_detector.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "matplotlib", "scipy", "tensorflow", "torch"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScreenDetector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=COMPAT_ARCH,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ScreenDetector",
)

app = BUNDLE(
    coll,
    name="ScreenDetector.app",
    icon="AppIcon.icns",
    bundle_identifier="com.screendetector.tool",
    info_plist={
        "CFBundleDisplayName": "框选屏幕检测工具",
        "CFBundleName": "ScreenDetector",
        "CFBundleShortVersionString": COMPAT_VERSION,
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "LSMinimumSystemVersion": COMPAT_MIN_SYS,
    },
    windowed=True,
)
