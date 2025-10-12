@echo off
setlocal
set PY=".venv\Scripts\python.exe"
if not exist %PY% (
  echo [ERROR] Could not find %PY%
  exit /b 1
)
echo [INFO] Current NumPy:
%PY% -c "import numpy,sys;print('numpy',numpy.__version__)"
echo [INFO] Forcing NumPy 1.x (ABI compatible)
%PY% -m pip uninstall -y numpy
%PY% -m pip install --only-binary=:all: numpy==1.26.4
echo [INFO] Final NumPy:
%PY% -c "import numpy,sys;print('numpy',numpy.__version__)"
echo [OK] Done. Launch FrameVision again.
