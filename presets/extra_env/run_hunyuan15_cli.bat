@echo off
setlocal EnableExtensions

REM Runs the Hunyuan 1.5 CLI using the local venv, with correct quoting.
REM Example:
REM   run_hunyuan15_cli.bat download --model 480p_t2v
REM   run_hunyuan15_cli.bat generate --model 480p_t2v --prompt "hello" --output out.mp4 --frames 61 --steps 30 --fps 15 --attn auto --offload --tiling

set "THIS_DIR=%~dp0"
for %%I in ("%THIS_DIR%..") do set "ROOT=%%~fI"
for %%I in ("%ROOT%\..") do set "ROOT=%%~fI"

set "VENV_PY=%ROOT%\.hunyuan15_env\Scripts\python.exe"
set "CLI_PY=%ROOT%\helpers\hunyuan15_cli.py"

if not exist "%VENV_PY%" (
  echo [ERROR] Venv python not found: "%VENV_PY%"
  echo         Run: "%ROOT%\presets\extra_env\hunuyan15_install.bat"
  exit /b 1
)

"%VENV_PY%" "%CLI_PY%" %*
exit /b %errorlevel%
