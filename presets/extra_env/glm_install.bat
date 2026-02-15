@echo off
setlocal enabledelayedexpansion

REM FrameVision optional install: GLM-Image (4bit)
REM Creates venv at: environments\.glm_env
REM Uses models folder at: models\ (HF cache is set by the worker)

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

cd /d "%ROOT%" || goto ERR

echo.
echo === GLM-Image (4bit) install ===
echo Root: %ROOT%

REM Try to find FrameVision-bundled python first; fallback to PATH python.
set "PY_EXE="
if exist "%ROOT%python\python.exe" set "PY_EXE=%ROOT%python\python.exe"
if exist "%ROOT%Python\python.exe" set "PY_EXE=%ROOT%Python\python.exe"
if exist "%ROOT%runtime\python.exe" set "PY_EXE=%ROOT%runtime\python.exe"
if "%PY_EXE%"=="" set "PY_EXE=python"

set "ENV_DIR=%ROOT%environments\.glm_env"

if exist "%ENV_DIR%\Scripts\python.exe" (
  echo.
  echo Environment already exists: %ENV_DIR%
) else (
  echo.
  echo Creating virtual environment...
  "%PY_EXE%" -m venv "%ENV_DIR%" || goto ERR
)

echo.
echo Activating environment...
call "%ENV_DIR%\Scripts\activate.bat" || goto ERR

echo.
echo Upgrading pip...
python -m pip install --upgrade pip || goto ERR

echo.
echo Installing dependencies...
pip install typing-extensions || goto ERR

REM NOTE: This matches the original installer (CUDA 12.6). If the user's system
REM doesn't support this, they can swap to a compatible torch build.
pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto ERR

pip install git+https://github.com/huggingface/diffusers.git || goto ERR
pip install --upgrade git+https://github.com/huggingface/transformers.git || goto ERR
pip install --upgrade sdnq accelerate || goto ERR

REM Gradio isn't required for the FrameVision pane, but some users might want it.
pip install --upgrade gradio || echo (warning) gradio install failed; continuing.

REM Optional speed-up (safe to fail)
pip install --upgrade triton-windows || echo (warning) triton-windows install failed; continuing without it.



echo.
echo Prefetching GLM-Image SDNQ 4bit model into models\hub ...
python "%SCRIPT_DIR%glm_downloads.py" --framevision-root "%ROOT%" || echo (warning) model prefetch failed; you can still download on first run.
echo.
echo ✅ Installation complete.
echo The GLM tool will now use: environments\.glm_env\Scripts\python.exe
pause
goto EOF

:ERR
echo.
echo ❌ There was an error during installation.
pause

:EOF
endlocal
