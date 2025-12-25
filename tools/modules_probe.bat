\
@echo off
setlocal
echo FrameVision Modules Probe (root-aware)
echo ============================================================
set PY=python
if exist "%~dp0..\ .venv\Scripts\python.exe" set PY="%~dp0..\ .venv\Scripts\python.exe"
pushd "%~dp0"
%PY% modules_probe.py
popd
echo.
pause
