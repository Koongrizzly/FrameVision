\
@echo off
setlocal
echo FrameVision Theme Probe (root-aware)
echo =====================================================
set PY=python
if exist "%~dp0..\ .venv\Scripts\python.exe" set PY="%~dp0..\ .venv\Scripts\python.exe"
pushd "%~dp0"
%PY% theme_probe.py
popd
echo.
pause
