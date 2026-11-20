#!/usr/bin/env bash
# ============================================================
#  Push this repository to GitHub (bash / git-bash / macOS / Linux).
#  Usage: from anywhere, run   bash scripts/push.sh
#  Edit REMOTE / BRANCH below if your repo differs.
# ============================================================
set -e
REMOTE="https://github.com/taabishahmad/text2sql-confidence.git"
BRANCH="main"

# move to the repository root (parent of this scripts/ folder)
cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  echo "[1/4] Initializing new git repository..."
  git init
  git branch -M "$BRANCH"
  git remote add origin "$REMOTE"
else
  echo "[1/4] Existing git repository detected."
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
fi

echo "[2/4] Staging all files..."
git add -A

echo "[3/4] Committing..."
git commit -m "Update code, data, figures and results for Certified Abstention (text-to-SQL)" || echo "   (nothing to commit)"

echo "[4/4] Pushing to $REMOTE ($BRANCH)..."
git push -u origin "$BRANCH"

echo "Done."
