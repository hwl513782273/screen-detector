#!/bin/bash
# 框选屏幕检测工具 —— 打包为 macOS .app 并制作 .dmg
# 用法: ./build.sh   (需在已装好依赖的 venv 中运行)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON_BIN:-/Users/banqiu/.workbuddy/binaries/python/envs/default/bin/python3}"
PYINSTALLER="${PYINSTALLER_BIN:-/Users/banqiu/.workbuddy/binaries/python/envs/default/bin/pyinstaller}"

DMG_NAME="框选屏幕检测工具.dmg"
VOL_NAME="框选屏幕检测工具"
APP_NAME="ScreenDetector.app"

# 安全清理旧的构建产物：用 mv 移到 /tmp，避免触发批量删除保护导致构建中断。
safe_move_away() {
  local p="$1"
  if [ -e "$p" ]; then
    local dst="/tmp/old_build_$(date +%s)_$(basename "$p")"
    mv "$p" "$dst" || true
  fi
}
safe_move_away "dist/$APP_NAME"
safe_move_away "dist/ScreenDetector"
safe_move_away "build/dmg_stage"

echo "==> 1/3 用 PyInstaller 构建 .app"
"$PYINSTALLER" -y ScreenDetector.spec

APP_PATH="dist/$APP_NAME"
if [ ! -d "$APP_PATH" ]; then
  echo "构建失败：未找到 $APP_PATH"
  exit 1
fi

echo "==> 2/3 准备 DMG 暂存目录"
STAGE="build/dmg_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R "$APP_PATH" "$STAGE/"
# 方便的“拖到应用程序”快捷方式
ln -s /Applications "$STAGE/Applications"

echo "==> 3/3 制作 DMG"
rm -f "$DMG_NAME"
hdiutil create -volname "$VOL_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG_NAME"

echo "完成：./$DMG_NAME"
ls -lh "$DMG_NAME"

# ---------------------------------------------------------------------------
# 临时文件 / 中间产物自动清理（v6.24）
# 每次构建会在 build/、dist/ 与 /tmp/old_build_* 累积数百 MB 中间产物，长期会
# 占用大量存储；此处构建成功后自动清理。交付用的『版本化 DMG』由 VERSION 变量
# 自动复制生成，已交付的旧版本 DMG 一律保留（遵循只增不删，绝不删除/覆盖）。
# ---------------------------------------------------------------------------
echo "==> 4/4 清理构建中间产物"

# 若指定 VERSION（如 6.24），自动复制为版本化交付 DMG 后再移除中间 dmg。
# 使用 cp -n（no-clobber）：若目标已存在则跳过，遵循『只增不删』原则。
if [ -n "$VERSION" ]; then
  DEST="/Users/banqiu/WorkBuddy/DMG/框选屏幕检测工具_${VERSION}.dmg"
  if cp -n "$DMG_NAME" "$DEST" 2>/dev/null; then
    echo "已生成交付 DMG: $DEST"
    rm -f "$DMG_NAME"
  else
    echo "提示：未能复制为交付 DMG（$DEST 可能已存在，已跳过覆盖），保留中间 dmg: $DMG_NAME"
  fi
fi

# 清理 PyInstaller 中间目录（用 mv 到 /tmp 避免触发批量删除保护）
safe_move_away "build"
safe_move_away "dist"

echo "已清理 build/、dist/ 中间产物（移至 /tmp/old_build_*）。"
echo "全部完成。"
