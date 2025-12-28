@echo off
setlocal EnableExtensions

REM Runs the Hunyuan 1.5 PySide6 UI using the local venv, with correct quoting.

set "THIS_DIR=%~dp0"
for %%I in ("%THIS_DIR%..") do set "ROOT=%%~fI"
for %%I in ("%ROOT%\..") do set "ROOT=%%~fI"

set "VENV_PY=%ROOT%\.hunyuan15_env\Scripts\python.exe"
set "UI_PY=%ROOT%\helpers\hunyuan15.py"

if not exist "%VENV_PY%" (
  echo [ERROR] Venv python not found: "%VENV_PY%"
  echo         Run: "%ROOT%\presets\extra_env\hunuyan15_install.bat"
  exit /b 1
)

"%VENV_PY%" "%UI_PY%"
exit /b %errorlevel%
