@echo off
setlocal EnableExtensions

REM Runs from: root\presets\extra_env\
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
pushd "%ROOT%" >nul

echo Installing FlashAttention 2 optional...
echo Note: On Windows this often requires Microsoft C++ Build Tools. If missing, this script will skip.
echo.

if not exist "environments\.qwen3tts\Scripts\python.exe" (
  echo ERROR: Environment not found at environments\.qwen3tts
  echo Please run presets\extra_env\install_qwentts.bat first.
  popd >nul
  pause
  exit /b 1
)

call "environments\.qwen3tts\Scripts\activate.bat" >nul 2>nul

python -c "import shutil,sys; sys.exit(0 if shutil.which('cl') else 1)" >nul 2>nul
if errorlevel 1 (
  echo Microsoft C++ Build Tools compiler cl.exe not found.
  echo Skipping FlashAttention install.
  echo You can still run without FlashAttention.
  popd >nul
  pause
  exit /b 0
)

echo Compiler detected. Attempting install...
python -m pip install -U pip setuptools wheel >nul 2>nul
python -m pip install flash-attn --no-build-isolation
if errorlevel 1 (
  echo FlashAttention install failed. Continuing without it.
  popd >nul
  pause
  exit /b 0
)

echo FlashAttention installed successfully.

popd >nul
pause
exit /b 0
