@echo off
setlocal EnableExtensions

REM ------------------------------------------------------------------
REM SongGeneration one-click installer (Windows)
REM Root = app root (two levels up from presets\extra_env)
REM venv:   .song_g_env\
REM repo:   models\song_generation\
REM output: output\music\song_g\
REM Prefers CUDA torch when NVIDIA is present (cu124 -> cu126 -> cu118 -> CPU)
REM ------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

echo.
echo [SongGeneration] Root: "%ROOT%"
echo.

pushd "%ROOT%"
if errorlevel 1 goto ROOT_FAIL

echo [SongGeneration] Step: folders
if not exist ".song_g_env" mkdir ".song_g_env"
if not exist "models" mkdir "models"
if not exist "output\music\song_g" mkdir "output\music\song_g"

echo [SongGeneration] Step: git
where git >nul 2>&1
if errorlevel 1 goto NO_GIT

if exist "models\song_generation\.git" (
  echo [SongGeneration] Updating existing repo...
  git -C "models\song_generation" pull
  if errorlevel 1 goto GIT_FAIL
) else (
  echo [SongGeneration] Cloning repo...
  if exist "models\song_generation" rmdir /s /q "models\song_generation" >nul 2>&1
  git clone --depth 1 "https://github.com/tencent-ailab/SongGeneration" "models\song_generation"
  if errorlevel 1 goto GIT_FAIL
)

echo [SongGeneration] Step: python detection
where py >nul 2>&1
if errorlevel 1 goto NO_PY

set "PY_CMD=py -3.10"
%PY_CMD% -V >nul 2>&1
if errorlevel 1 goto TRY_PY3
goto PY_OK

:TRY_PY3
set "PY_CMD=py -3"
%PY_CMD% -V >nul 2>&1
if errorlevel 1 goto NO_PYVER

:PY_OK
echo [SongGeneration] Using: %PY_CMD%

echo [SongGeneration] Step: venv
if exist ".song_g_env\Scripts\python.exe" goto VENV_OK

echo [SongGeneration] Creating venv...
%PY_CMD% -m venv ".song_g_env"
if errorlevel 1 goto VENV_FAIL

:VENV_OK
call ".song_g_env\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel

echo [SongGeneration] Step: torch
echo [SongGeneration] Removing old torch packages if any...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

where nvidia-smi >nul 2>&1
if errorlevel 1 goto TORCH_CPU

echo [SongGeneration] NVIDIA detected. Installing PyTorch 2.6.0 CUDA 12.4 cu124...
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto TORCH_TRY_CU126
goto TORCH_DONE

:TORCH_TRY_CU126
echo [SongGeneration] cu124 failed. Trying CUDA 12.6 cu126...
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto TORCH_TRY_CU118
goto TORCH_DONE

:TORCH_TRY_CU118
echo [SongGeneration] cu126 failed. Trying CUDA 11.8 cu118...
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto TORCH_CPU
goto TORCH_DONE

:TORCH_CPU
echo [SongGeneration] Installing CPU torch...
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

:TORCH_DONE
echo [SongGeneration] Step: deps

if exist "presets\extra_env\SG_req.txt" (
  echo [SongGeneration] Installing base deps from SG_req.txt ...
  pip install --upgrade -r "presets\extra_env\SG_req.txt"
) else (
  echo [SongGeneration] WARNING: SG_req.txt not found.
)

if exist "models\song_generation\requirements.txt" (
  echo [SongGeneration] Installing repo requirements.txt ...
  pip install -r "models\song_generation\requirements.txt"
)

if exist "models\song_generation\requirements_nodeps.txt" (
  echo [SongGeneration] Installing requirements_nodeps.txt no-deps ...
  pip install -r "models\song_generation\requirements_nodeps.txt" --no-deps
)

echo [SongGeneration] Step: py311 hydra fix
python -m pip install --upgrade "hydra-core>=1.3.2" "omegaconf>=2.3.0"

echo.
echo [SongGeneration] DONE.
echo.
pause
goto END

:ROOT_FAIL
echo [ERROR] Could not cd to root: "%ROOT%"
goto END

:NO_GIT
echo [ERROR] Git not found in PATH. Install Git for Windows then rerun.
goto END

:GIT_FAIL
echo [ERROR] Git operation failed.
goto END

:NO_PY
echo [ERROR] Python launcher py not found. Install Python then rerun.
goto END

:NO_PYVER
echo [ERROR] No usable Python found via py. Need Python 3.x.
goto END

:VENV_FAIL
echo [ERROR] venv creation failed.
goto END

:END
popd
endlocal
