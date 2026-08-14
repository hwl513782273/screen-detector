#!/usr/bin/env bash
# 一键发布脚本：把当前已测试通过的版本提交并打标签
# 用法:
#   bash git_release.sh <版本号> [一句话说明]
# 例:
#   bash git_release.sh 6.48 "匹配模式增加仅变化检测"
#   bash git_release.sh 6.49
set -e
VER="$1"
NOTE="${2:-}"
if [ -z "$VER" ]; then
  echo "用法: bash git_release.sh <版本号> [一句话说明]"
  exit 1
fi
cd "$(dirname "$0")"

# 1) 暂存所有改动（.gitignore 已排除缓存/构建产物/旧报告）
git add -A

# 2) 没有改动则跳过
if git diff --cached --quiet; then
  echo "没有待提交的改动，跳过。"
  exit 0
fi

# 3) 提交 + 打标签
MSG="v${VER}：${NOTE}

（由 git_release.sh 自动提交；需先通过功能自检后再发布）"
git commit -q -m "$MSG"
git tag -a "v${VER}" -m "v${VER}"

echo "✅ 已提交并打标签 v${VER}"
git log --oneline -1
echo "--- 最近标签 ---"
git tag -l | tail -5

# 4) 若已配置远程，提示如何推送（本脚本不自动 push，避免误推）
if git remote -v | grep -q .; then
  echo
  echo "已检测到远程仓库。需要上线时执行："
  echo "  git push origin main --tags"
else
  echo
  echo "尚未配置远程。配置并推送："
  echo "  git remote add origin <你的仓库URL>"
  echo "  git push origin main --tags"
fi
