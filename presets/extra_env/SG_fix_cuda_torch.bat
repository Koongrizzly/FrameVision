@echo off
setlocal EnableExtensions

REM ------------------------------------------------------------------
REM SongGeneration - switch existing .song_g_env to CUDA-enabled PyTorch
REM Tries CUDA 12.4 -> CUDA 12.6 -> CUDA 11.8 -> CPU fallback
REM Uses the official "previous versions" commands for torch 2.6.0.
REM ------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

if not exist ".song_g_env\Scripts\python.exe" (
  echo [SG CUDA] Env not found: .song_g_env
  echo Run SG_install.bat first.
  popd
  exit /b 1
)

call ".song_g_env\Scripts\activate.bat"

where nvidia-smi >nul 2>&1
if errorlevel 1 goto NO_NVIDIA

echo [SG CUDA] NVIDIA detected.
echo [SG CUDA] Removing old torch packages if any...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo [SG CUDA] Installing PyTorch 2.6.0 CUDA 12.4 cu124...
python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto TRY_CU126
goto VERIFY

:TRY_CU126
echo [SG CUDA] cu124 failed. Trying CUDA 12.6 cu126...
python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto TRY_CU118
goto VERIFY

:TRY_CU118
echo [SG CUDA] cu126 failed. Trying CUDA 11.8 cu118...
python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto FALLBACK_CPU
goto VERIFY

:FALLBACK_CPU
echo [SG CUDA] All CUDA installs failed. Falling back to CPU torch...
python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
goto VERIFY

:NO_NVIDIA
echo [SG CUDA] NVIDIA not detected. Installing CPU torch...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
python -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
goto VERIFY

:VERIFY
echo.
echo [SG CUDA] Verify:
python -c "import torch; print('torch',torch.__version__); print('cuda',torch.cuda.is_available()); print('cuda_ver',torch.version.cuda)"
echo.
echo [SG CUDA] Done.
pause

popd
endlocal
