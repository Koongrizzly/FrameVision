@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ------------------------------------------------------------
REM FrameVision Optional Install: Whisper (Faster-Whisper)
REM Creates an isolated venv at:  <root>\environments\.whisper\
REM Installs requirements, then downloads the default model.
REM ------------------------------------------------------------

REM Resolve project root (this .bat lives in: presets\extra_env\)
set "BAT_DIR=%~dp0"
for %%I in ("%BAT_DIR%..\..") do set "ROOT=%%~fI"

set "ENV_DIR=%ROOT%\environments\.whisper"
set "REQ_FILE=%ROOT%\presets\extra_env\whisper_req.txt"
set "DL_SCRIPT=%ROOT%\presets\extra_env\whisper_download_models.py"

echo.
echo [Whisper] Root: %ROOT%
echo [Whisper] Env : %ENV_DIR%
echo.

REM Find a usable Python
set "PY="
if exist "%ROOT%\python\python.exe" set "PY=%ROOT%\python\python.exe"
if exist "%ROOT%\Python\python.exe" set "PY=%ROOT%\Python\python.exe"
if exist "%ROOT%\presets\python\python.exe" set "PY=%ROOT%\presets\python\python.exe"
if "%PY%"=="" set "PY=python"

echo [Whisper] Using Python: %PY%
%PY% --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo [Whisper] ERROR: Python not found.
  echo - Install Python or ensure it is on PATH.
  echo - Or place a portable python at ^<root^>\python\python.exe
  echo.
  pause
  exit /b 1
)

REM Fresh restart of the venv (delete & recreate)
if exist "%ENV_DIR%\Scripts\python.exe" (
  echo [Whisper] Existing env detected, removing for a clean reinstall...
  rmdir /s /q "%ENV_DIR%"
)

echo [Whisper] Creating venv...
%PY% -m venv "%ENV_DIR%"
if errorlevel 1 (
  echo.
  echo [Whisper] ERROR: venv creation failed.
  pause
  exit /b 1
)

set "VENV_PY=%ENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo.
  echo [Whisper] ERROR: venv python not found: %VENV_PY%
  pause
  exit /b 1
)

echo [Whisper] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo.
  echo [Whisper] ERROR: pip upgrade failed.
  pause
  exit /b 1
)

echo [Whisper] Installing requirements...
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
  echo.
  echo [Whisper] ERROR: pip install failed.
  pause
  exit /b 1
)

echo.
echo [Whisper] Downloading default model (medium)...
"%VENV_PY%" "%DL_SCRIPT%" --model medium
if errorlevel 1 (
  echo.
  echo [Whisper] ERROR: model download failed.
  pause
  exit /b 1
)

echo.
echo [Whisper] SUCCESS.
echo - Env: %ENV_DIR%
echo - Model: %ROOT%\models\faster_whisper\medium\
echo.
pause
exit /b 0
