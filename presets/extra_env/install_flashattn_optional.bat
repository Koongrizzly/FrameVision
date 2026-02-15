@echo off
setlocal EnableExtensions

REM Runs from: root\presets\extra_env\
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
pushd "%ROOT%" >nul

echo Installing FlashAttention optional (Windows wheels)...
echo This installer:
echo   - Pins Torch to CUDA 12.9 (Torch 2.8.0)
echo   - Installs a prebuilt flash_attn wheel that matches your Python version

echo.

REM --- Hard requirement: Qwen TTS env must exist ---
if not exist "environments\.qwen3tts\Scripts\python.exe" (
  echo ERROR: Environment not found at environments\.qwen3tts
  echo Please run presets\extra_env\install_qwentts.bat first.
  popd >nul
  pause
  exit /b 1
)

call "environments\.qwen3tts\Scripts\activate.bat" >nul 2>nul

echo Upgrading pip/setuptools/wheel...
python -m pip install -U pip setuptools wheel
if errorlevel 1 echo WARN: pip upgrade failed. Continuing anyway...

echo.
echo Detecting Python version in this environment...
set "PYVER="
for /f "tokens=2 delims= " %%V in ('python -V 2^>^&1') do set "PYVER=%%V"

set "PYMAJ="
set "PYMIN="
for /f "tokens=1,2 delims=." %%A in ("%PYVER%") do (
  set "PYMAJ=%%A"
  set "PYMIN=%%B"
)

echo Using Python %PYVER%

if not "%PYMAJ%"=="3" (
  echo ERROR: Unsupported Python version: %PYVER%
  echo FlashAttention wheels are available only for Python 3.x.
  popd >nul
  pause
  exit /b 1
)

REM Choose wheel URL based on Python minor version
set "WHEEL_URL="
if "%PYMIN%"=="10" set "WHEEL_URL=https://github.com/gjnave/support-files/raw/main/support/wheels/py310/flash_attn-2.8.2-cp310-cp310-win_amd64.whl"
if "%PYMIN%"=="11" set "WHEEL_URL=https://github.com/gjnave/support-files/raw/main/support/wheels/py311/flash_attn-2.8.2-cp311-cp311-win_amd64.whl"
if "%PYMIN%"=="12" set "WHEEL_URL=https://github.com/gjnave/support-files/raw/main/support/wheels/py312/flash_attn-2.8.2-cp312-cp312-win_amd64.whl"
if "%PYMIN%"=="13" set "WHEEL_URL=https://github.com/gjnave/support-files/raw/main/support/wheels/py313/flash_attn-2.8.2-cp313-cp313-win_amd64.whl"

if "%WHEEL_URL%"=="" (
  echo ERROR: No prebuilt FlashAttention wheel configured for Python %PYVER%.
  echo Supported: 3.10, 3.11, 3.12, 3.13
  popd >nul
  pause
  exit /b 1
)

echo.
echo Installing compatibility dependencies...
python -m pip install typing_extensions==4.12.2
python -m pip install "triton-windows<3.5"

echo.
echo Installing Torch 2.8.0 (CUDA 12.9)...
python -m pip install --upgrade --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
if errorlevel 1 (
  echo ERROR: Torch install failed.
  echo Without a compatible Torch build, FlashAttention will not work.
  popd >nul
  pause
  exit /b 1
)

echo.
echo Installing FlashAttention wheel...
python -m pip install "%WHEEL_URL%"
if errorlevel 1 (
  echo.
  echo Wheel install failed.
  echo Trying fallback build ^(requires Microsoft C++ Build Tools^)...

  where cl >nul 2>nul
  if errorlevel 1 (
    echo Microsoft C++ Build Tools compiler cl.exe not found.
    echo Skipping FlashAttention install.
    echo You can still run without FlashAttention.
    popd >nul
    pause
    exit /b 0
  )

  echo Compiler detected. Attempting source install...
  python -m pip install flash-attn --no-build-isolation
  if errorlevel 1 (
    echo FlashAttention install failed. Continuing without it.
    popd >nul
    pause
    exit /b 0
  )
)

echo.
echo FlashAttention installed successfully.

popd >nul
pause
exit /b 0
