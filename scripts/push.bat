@echo off
setlocal
REM ============================================================
REM  Push this repository to GitHub (Windows CMD).
REM  Usage: from anywhere, run   scripts\push.bat
REM  Edit REMOTE / BRANCH below if your repo differs.
REM ============================================================
set "REMOTE=https://github.com/taabishahmad/text2sql-confidence.git"
set "BRANCH=main"

REM move to the repository root (parent of this scripts\ folder)
cd /d "%~dp0.."

if not exist ".git" (
    echo [1/4] Initializing new git repository...
    git init
    git branch -M %BRANCH%
    git remote add origin %REMOTE%
) else (
    echo [1/4] Existing git repository detected.
    git remote get-url origin >nul 2>&1 || git remote add origin %REMOTE%
)

echo [2/4] Staging all files...
git add -A

echo [3/4] Committing...
git commit -m "Update code, data, figures and results for Certified Abstention (text-to-SQL)" || echo    (nothing to commit)

echo [4/4] Pushing to %REMOTE% (%BRANCH%)...
git push -u origin %BRANCH%

echo.
echo Done. If prompted, sign in to GitHub in the browser / credential window.
pause
