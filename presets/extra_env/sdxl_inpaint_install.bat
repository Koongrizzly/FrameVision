@echo off
setlocal ENABLEDELAYEDEXPANSION

REM SDXL Inpaint installer launcher.
REM Actual installer logic lives in sdxl_inpaint_install.py so path handling stays sane.
REM Creates/updates: <project_root>\environments\.sdxl_inpaint

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "INSTALL_PY=%SCRIPT_DIR%sdxl_inpaint_install.py"

set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "%INSTALL_PY%" (
  echo ERROR: Python installer not found:
  echo   "%INSTALL_PY%"
  pause
  exit /b 1
)

echo Running SDXL Inpaint Python installer...
"%PY%" "%INSTALL_PY%"
if errorlevel 1 (
  echo.
  echo ERROR: SDXL Inpaint installer failed.
  pause
  exit /b 1
)

echo.
echo Done.
exit /b 0
