@echo off
setlocal ENABLEDELAYEDEXPANSION

REM SDXL Inpaint extra environment installer
REM Creates/updates: <project_root>\.sdxl_inpaint
REM Installs requirements from: <project_root>\presets\extra_env\sdxl_inpaint_req.txt
REM Then runs: <project_root>\scripts\background_download.py

REM Resolve project root from this script location: <root>\presets\extra_env\sdxl_inpaint_install.bat
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

set "MAIN_PY=%ROOT%\.venv\Scripts\python.exe"
set "ENV_DIR=%ROOT%\.sdxl_inpaint"
set "ENV_PY=%ENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%ROOT%\presets\extra_env\sdxl_inpaint_req.txt"
set "BG_DL=%ROOT%\scripts\background_download.py"

echo.
echo ===============================
echo  SDXL Inpaint Env Installer
echo ===============================
echo Root: "%ROOT%"
echo Env : "%ENV_DIR%"
echo.

if not exist "%MAIN_PY%" (
  echo ERROR: Main venv python not found:
  echo   "%MAIN_PY%"
  echo This installer expects FrameVision's main .venv to exist.
  echo.
  pause
  exit /b 1
)

if not exist "%REQ_FILE%" (
  echo ERROR: Requirements file not found:
  echo   "%REQ_FILE%"
  echo.
  pause
  exit /b 1
)

REM Create env if missing
if not exist "%ENV_PY%" (
  echo Creating isolated SDXL inpaint env...
  "%MAIN_PY%" -m venv "%ENV_DIR%"
  if errorlevel 1 (
    echo ERROR: Failed to create venv at "%ENV_DIR%"
    pause
    exit /b 1
  )
) else (
  echo Env already exists. Updating packages...
)

echo Upgrading pip tooling...
"%ENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo ERROR: Failed to upgrade pip tooling.
  pause
  exit /b 1
)

REM Install a known-good Torch build for SDXL Inpaint.
REM If you want to change CUDA build, edit the index-url below.
echo Installing Torch (cu124)...
"%ENV_PY%" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
if errorlevel 1 (
  echo ERROR: Torch install failed.
  echo Tip: If you need a different CUDA build, edit this .bat and adjust the index-url / versions.
  pause
  exit /b 1
)

echo Installing SDXL-inpaint requirements...
"%ENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
  echo ERROR: Requirements install failed.
  pause
  exit /b 1
)

echo.
echo Running background downloads...
if exist "%BG_DL%" (
  "%MAIN_PY%" "%BG_DL%"
  if errorlevel 1 (
    echo WARNING: background_download.py returned an error.
    echo You can run it again later from the app.
  )
) else (
  echo WARNING: background_download.py not found at:
  echo   "%BG_DL%"
)

echo.
echo Done.
echo.
exit /b 0
