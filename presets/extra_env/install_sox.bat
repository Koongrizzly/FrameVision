@echo off
setlocal EnableExtensions

REM Runs from: root\presets\extra_env\
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
pushd "%ROOT%" >nul

echo.
echo [Qwen3-TTS] Installing SoX (required)
echo ----------------------------------------------------------
echo Using static-sox inside the venv (no system-wide install).
echo.

if not exist "environments\.qwen3tts\Scripts\python.exe" (
  echo ERROR: Environment not found at environments\.qwen3tts
  echo Please run presets\extra_env\install_qwentts.bat first.
  popd >nul
  exit /b 1
)

call "environments\.qwen3tts\Scripts\activate.bat" >nul 2>nul

python -m pip install -U static-sox >nul 2>nul

REM Verify sox is resolvable after adding paths
for /f "delims=" %%A in ('python -c "import static_sox,shutil; static_sox.add_paths(weak=True); p=shutil.which(\"sox\"); print(p if p else \"\")"') do set "SOXPATH=%%A"

if "%SOXPATH%"=="" (
  echo ERROR: sox.exe still not found after static-sox setup.
  popd >nul
  exit /b 1
)

echo OK: sox found at %SOXPATH%

popd >nul
exit /b 0
