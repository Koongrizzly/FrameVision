@echo off
setlocal EnableExtensions

REM Fix for Python 3.11 + old hydra-core:
REM ValueError: mutable default ... OverrideDirname ... use default_factory

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

if not exist ".song_g_env\Scripts\python.exe" (
  echo [SG FIX] Env not found: .song_g_env
  echo Run SG_install.bat first.
  popd
  exit /b 1
)

call ".song_g_env\Scripts\activate.bat"

echo [SG FIX] Upgrading hydra-core and omegaconf for Python 3.11 compatibility...
python -m pip install --upgrade "hydra-core>=1.3.2" "omegaconf>=2.3.0"

echo [SG FIX] Done.
pause
popd
endlocal
