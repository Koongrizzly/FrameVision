@echo off
setlocal ENABLEDELAYEDEXPANSION

echo =========================================
echo   Z-image Turbo - environment installer
echo   - Creates: .zimage_env
echo   - Installs deps from: presets\extra_env\zimage_req.txt
echo   Note: This step does NOT download any model weights.
echo         Use the optional installs for:
echo           - Full FP16 model
echo           - Z-image Turbo GGUF (Q4/Q5/Q6/Q8)
echo =========================================
echo.

REM Get directory of this script (presets\extra_env)
set "SCRIPT_DIR=%~dp0"

REM Go to project root (two levels up)
cd /d "%SCRIPT_DIR%\..\.."
if errorlevel 1 (
  echo [ERROR] Could not cd to project root.
  pause
  exit /b 1
)

set "ROOT=%CD%"

REM Choose a bootstrap python (prefer FrameVision .venv if present)
set "BOOT_PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%BOOT_PY%" goto have_py

set "BOOT_PY=%ROOT%\.venv\bin\python"
if exist "%BOOT_PY%" goto have_py

set "BOOT_PY=python"

:have_py
echo [1/3] Using bootstrap python: %BOOT_PY%

REM Create venv if needed
if exist "%ROOT%\.zimage_env\Scripts\python.exe" goto venv_ok
if exist "%ROOT%\.zimage_env\bin\python" goto venv_ok

echo [2/3] Creating venv: %ROOT%\.zimage_env
"%BOOT_PY%" -m venv "%ROOT%\.zimage_env"
if errorlevel 1 (
  echo [ERROR] Failed to create .zimage_env
  pause
  exit /b 1
)

:venv_ok
set "PYTHON_EXEC=%ROOT%\.zimage_env\Scripts\python.exe"
if exist "%PYTHON_EXEC%" goto got_venv_py
set "PYTHON_EXEC=%ROOT%\.zimage_env\bin\python"

:got_venv_py
echo [3/3] Installing Python dependencies from presets\extra_env\zimage_req.txt ...
"%PYTHON_EXEC%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [WARN] pip upgrade failed (continuing)...
)

if not exist "presets\extra_env\zimage_req.txt" (
  echo [ERROR] Missing presets\extra_env\zimage_req.txt
  pause
  exit /b 1
)

"%PYTHON_EXEC%" -m pip install -r "presets\extra_env\zimage_req.txt"
if errorlevel 1 (
  echo [ERROR] Failed to install Python requirements.
  pause
  exit /b 1
)

echo.
echo =========================================
echo   Z-image environment install complete!
echo   Environment: .zimage_env
echo   Next: pick a model download option (FP16 or GGUF) in Optional Installs.
echo =========================================
echo.
pause
endlocal
