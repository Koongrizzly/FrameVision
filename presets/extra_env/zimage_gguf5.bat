@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ------------------------------------------------------------
REM FrameVision optional installer: Z-Image Turbo GGUF (Low VRAM)
REM Installs into: <root>\models\Z-Image-Turbo GGUF
REM ------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

set "TARGET=%ROOT%\models\Z-Image-Turbo GGUF"

echo.
echo [zimage-gguf] Root  : %ROOT%
echo [zimage-gguf] Target: %TARGET%
echo.

if not exist "%TARGET%" mkdir "%TARGET%"

REM Pick a Python (prefer FrameVision's z-image env if present)
set "PY="
if exist "%ROOT%\.zimage_env\Scripts\python.exe" set "PY=%ROOT%\.zimage_env\Scripts\python.exe"
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"

if not defined PY (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PY=%%P"
    goto :py_found
  )
)
:py_found

if not defined PY (
  echo [zimage-gguf] ERROR: could not find python.exe
  echo Please install Python or ensure it is on PATH.
  pause
  exit /b 1
)

echo [zimage-gguf] Using python: %PY%
echo.

REM Run the installer python (shipped next to this bat)
"%PY%" "%SCRIPT_DIR%zimage_gguf_install.py" "%ROOT%" "%TARGET%"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [zimage-gguf] ERROR: installer failed.
  echo Check logs inside: %TARGET%\_tmp
  pause
  exit /b %RC%
)

echo.
echo [zimage-gguf] Done. You can now pick the GGUF engine in FrameVision.
pause
exit /b 0
