#!/bin/bash
# 框选屏幕检测工具 —— 免打包直接运行（开发/调试用）
# 使用隔离 venv 中的 Python 运行，无需每次重新打包。
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON_BIN:-/Users/banqiu/.workbuddy/binaries/python/envs/default/bin/python3}"
if [ ! -x "$PY" ]; then
  echo "未找到隔离 Python，请先按文档创建 venv 并安装依赖。"
  exit 1
fi
cd "$SCRIPT_DIR"
exec "$PY" screen_detector.py
