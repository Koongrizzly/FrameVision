@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM This installer lives in presets\extra_env. Resolve the FrameVision root
REM from the installer location so it works regardless of the launch directory.
for %%R in ("%~dp0..\..") do set "FRAMEVISION_ROOT=%%~fR"
cd /d "%FRAMEVISION_ROOT%"
if errorlevel 1 (
  echo ERROR: Could not switch to the FrameVision root:
  echo   %FRAMEVISION_ROOT%
  pause
  exit /b 1
)
REM Clean legacy root copies only after the new canonical files are present.
if exist "%CD%\presets\download_models.bat" if exist "%CD%\download_models.bat" del /q "%CD%\download_models.bat" >nul 2>&1
if exist "%CD%\presets\requirements-runtime.txt" if exist "%CD%\requirements-runtime.txt" del /q "%CD%\requirements-runtime.txt" >nul 2>&1
if exist "%CD%\helpers\generate.py" if exist "%CD%\generate.py" del /q "%CD%\generate.py" >nul 2>&1
if exist "%CD%\helpers\generate_ref.py" if exist "%CD%\generate_ref.py" del /q "%CD%\generate_ref.py" >nul 2>&1
title MiniMax-H3 INT4 ConvRot Standalone Installer

echo ============================================================
echo MiniMax-H3 INT4 W4A8 ConvRot Standalone - CLEAN INSTALL
echo ============================================================
echo.
echo This creates an isolated Python environment and runtime.
echo After the runtime install finishes, the model downloader starts
echo automatically and lets you choose FL2VA, Ref2VA, both, and LoRAs.
echo.

REM ------------------------------------------------------------
REM Locate Conda robustly. Do not assume Conda was added to PATH.
REM Windows Conda installs may expose Scripts\conda.exe and/or
REM condabin\conda.bat. Miniforge and AppData installs are common.
REM ------------------------------------------------------------
set "CONDA_CMD="
set "CONDA_KIND="

REM 1) Reuse Conda's own environment variables when launched from a Conda prompt.
if defined CONDA_EXE if exist "%CONDA_EXE%" (
  set "CONDA_CMD=%CONDA_EXE%"
  set "CONDA_KIND=exe"
)
if not defined CONDA_CMD if defined CONDA_BAT if exist "%CONDA_BAT%" (
  set "CONDA_CMD=%CONDA_BAT%"
  set "CONDA_KIND=bat"
)
if not defined CONDA_CMD if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\Scripts\conda.exe" (
    set "CONDA_CMD=%CONDA_PREFIX%\Scripts\conda.exe"
    set "CONDA_KIND=exe"
  ) else if exist "%CONDA_PREFIX%\condabin\conda.bat" (
    set "CONDA_CMD=%CONDA_PREFIX%\condabin\conda.bat"
    set "CONDA_KIND=bat"
  )
)
if not defined CONDA_CMD if defined CONDA_PYTHON_EXE (
  for %%R in ("%CONDA_PYTHON_EXE%\..") do (
    if exist "%%~fR\Scripts\conda.exe" (
      set "CONDA_CMD=%%~fR\Scripts\conda.exe"
      set "CONDA_KIND=exe"
    ) else if exist "%%~fR\condabin\conda.bat" (
      set "CONDA_CMD=%%~fR\condabin\conda.bat"
      set "CONDA_KIND=bat"
    )
  )
)

REM 2) Check common per-user, AppData, all-users, Program Files, and Miniforge locations.
if not defined CONDA_CMD (
  for %%R in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\Miniforge3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\Miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\Anaconda3"
    "%LOCALAPPDATA%\miniforge3"
    "%LOCALAPPDATA%\Miniforge3"
    "%PROGRAMDATA%\miniconda3"
    "%PROGRAMDATA%\Miniconda3"
    "%PROGRAMDATA%\anaconda3"
    "%PROGRAMDATA%\Anaconda3"
    "%PROGRAMDATA%\miniforge3"
    "%PROGRAMDATA%\Miniforge3"
    "%ProgramFiles%\Miniconda3"
    "%ProgramFiles%\Anaconda3"
    "%ProgramFiles%\Miniforge3"
  ) do (
    if not defined CONDA_CMD if exist "%%~R\Scripts\conda.exe" (
      set "CONDA_CMD=%%~R\Scripts\conda.exe"
      set "CONDA_KIND=exe"
    )
    if not defined CONDA_CMD if exist "%%~R\condabin\conda.bat" (
      set "CONDA_CMD=%%~R\condabin\conda.bat"
      set "CONDA_KIND=bat"
    )
  )
)

REM 3) PATH lookup. Check both the executable and the Windows batch wrapper.
if not defined CONDA_CMD (
  for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA_CMD (
    set "CONDA_CMD=%%C"
    set "CONDA_KIND=exe"
  )
)
if not defined CONDA_CMD (
  for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_CMD (
    set "CONDA_CMD=%%C"
    set "CONDA_KIND=bat"
  )
)

REM 4) Last resort: ask whether the user has Conda in a custom location.
if not defined CONDA_CMD if /i "%MINIMAX_H3_GUI_INSTALL%"=="1" (
  echo ERROR: Conda was not found in the normal locations.
  echo Install Miniconda or Anaconda, then use the FrameVision MiniMax installer again.
  exit /b 1
)

if not defined CONDA_CMD (
  echo.
  echo Conda not found in default locations.
  echo If you have Conda installed in a custom location, continue with Y
  echo to enter the exact path. Choose N to exit and download + install
  echo Miniconda first.
  echo.
  choice /C YN /N /M "Do you want to enter a custom Conda path? [Y/N]: "
  if errorlevel 2 (
    echo.
    echo Installer cancelled. Please download and install Miniconda, then run this installer again.
    pause
    exit /b 1
  )

  echo.
  echo Enter the Conda installation folder, for example:
  echo   C:\Users\YourName\miniconda3
  echo   C:\Users\YourName\AppData\Local\anaconda3
  echo   D:\Miniforge3
  echo.
  set /p "CONDA_ROOT=Exact Conda folder: "
  if defined CONDA_ROOT (
    set "CONDA_ROOT=!CONDA_ROOT:"=!"
    if exist "!CONDA_ROOT!\Scripts\conda.exe" (
      set "CONDA_CMD=!CONDA_ROOT!\Scripts\conda.exe"
      set "CONDA_KIND=exe"
    ) else if exist "!CONDA_ROOT!\condabin\conda.bat" (
      set "CONDA_CMD=!CONDA_ROOT!\condabin\conda.bat"
      set "CONDA_KIND=bat"
    )
  )
)

if not defined CONDA_CMD (
  echo.
  echo ERROR: No working Conda installation was found at the path you entered.
  echo Check the path and run the installer again, or install Miniconda first.
  pause
  exit /b 1
)

echo Found Conda: %CONDA_CMD%

REM Validate the discovered command before using it.
if /i "%CONDA_KIND%"=="bat" (
  call "%CONDA_CMD%" --version >nul 2>&1
) else (
  "%CONDA_CMD%" --version >nul 2>&1
)
if errorlevel 1 (
  echo ERROR: Conda was found but could not be executed:
  echo   %CONDA_CMD%
  pause
  exit /b 1
)

set "ENV=%CD%\environments\.minimax_h3_int4"
set "PY=%ENV%\python.exe"
if not exist "%PY%" (
  echo [1/8] Creating isolated Python 3.12 environment...
  if /i "%CONDA_KIND%"=="bat" (
    call "%CONDA_CMD%" create -y -p "%ENV%" python=3.12 pip
  ) else (
    "%CONDA_CMD%" create -y -p "%ENV%" python=3.12 pip
  )
  if errorlevel 1 goto :fail
) else (
  echo [1/8] Environment already exists.
)

echo [2/8] Updating pip...
"%PY%" -m pip install --no-warn-script-location --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/8] Installing matched PyTorch 2.11.0 CUDA 13.0 stack...
echo       torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0
REM PyTorch 2.12.1 does not publish a matching torchaudio 2.12.1 cu130 Windows wheel.
REM Clear any partial packages from an interrupted/failed previous install first.
"%PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
"%PY%" -m pip install --no-warn-script-location torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fail

echo [4/8] Installing Ref2VA audio support...
echo       TorchCodec 0.11.1 + private FFmpeg 7.1 shared libraries.
echo       This does NOT install FFmpeg through Conda and does NOT modify the GUI/runtime code.
set "TC_FFMPEG=%ENV%\torchcodec_ffmpeg"
set "TC_FFMPEG_ZIP=%TEMP%\minimax_ffmpeg_shared_7_1.zip"
set "TC_FFMPEG_TMP=%ENV%\torchcodec_ffmpeg_extract"
set "TC_FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip"

REM TorchCodec 0.11.x is the matching line for Torch 2.11. Use --no-deps so the
REM already matched Torch/CUDA installation cannot be replaced.
"%PY%" -m pip install --no-warn-script-location --no-deps "torchcodec==0.11.1"
if errorlevel 1 goto :fail

REM TorchCodec needs FFmpeg shared DLLs on Windows. Keep a private copy inside
REM the isolated MiniMax environment instead of adding Conda/GTK packages.
if not exist "%TC_FFMPEG%\bin\avcodec-61.dll" (
  echo       Downloading FFmpeg 7.1 shared build...
  where curl.exe >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Windows curl.exe was not found.
    goto :fail
  )
  curl.exe -L --fail --retry 5 --retry-delay 2 --continue-at - -o "%TC_FFMPEG_ZIP%" "%TC_FFMPEG_URL%"
  if errorlevel 1 goto :fail
  if exist "%TC_FFMPEG_TMP%" rmdir /s /q "%TC_FFMPEG_TMP%"
  mkdir "%TC_FFMPEG_TMP%" >nul 2>&1
  "%PY%" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "%TC_FFMPEG_ZIP%" "%TC_FFMPEG_TMP%"
  if errorlevel 1 goto :fail
  if exist "%TC_FFMPEG%" rmdir /s /q "%TC_FFMPEG%"
  "%PY%" -c "import pathlib,shutil,sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); roots=[p for p in src.iterdir() if p.is_dir()]; assert roots, 'FFmpeg archive had no folder'; shutil.move(str(roots[0]), str(dst))" "%TC_FFMPEG_TMP%" "%TC_FFMPEG%"
  if errorlevel 1 goto :fail
  rmdir /s /q "%TC_FFMPEG_TMP%" >nul 2>&1
  del /q "%TC_FFMPEG_ZIP%" >nul 2>&1
) else (
  echo       Private FFmpeg 7.1 shared libraries already present; skipping download.
)

REM Make the private FFmpeg DLL folder visible automatically whenever this
REM isolated Python starts, including when the GUI launches python.exe directly.
> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo import os
>> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo from pathlib import Path
>> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo _p = str(Path(__file__).resolve().parents[2] / "torchcodec_ffmpeg" / "bin")
>> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo if os.path.isdir(_p):
>> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo     os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
>> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.py" echo     _dll_dir = os.add_dll_directory(_p) if hasattr(os, "add_dll_directory") else None
> "%ENV%\Lib\site-packages\minimax_ffmpeg_dll_path.pth" echo import minimax_ffmpeg_dll_path

REM Exercise the same torchaudio.load path used by Ref2VA before continuing.
echo       Running Ref2VA audio decode smoke test...
"%PY%" -c "import os,tempfile,wave,torchaudio,torchcodec; p=tempfile.mktemp(suffix='.wav'); w=wave.open(p,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(32000); w.writeframes(b'\x00\x00'*3200); w.close(); x,sr=torchaudio.load(p); os.remove(p); assert sr==32000 and x.shape[-1]==3200, (sr,x.shape); print('TorchCodec Ref2VA audio OK:', torchcodec.__version__, sr, tuple(x.shape))"
if errorlevel 1 goto :fail

echo [5/8] Installing MiniMax-H3 runtime dependencies...
"%PY%" -m pip install --no-warn-script-location -r "%CD%\presets\requirements-runtime.txt"
if errorlevel 1 goto :fail

echo [6/8] Installing Triton-Windows for Torch 2.10+...
REM SageAttention imports Triton at runtime. Triton-Windows 3.6 is the compatible
REM Windows line for PyTorch 2.10+; cap below 3.7 to avoid a future incompatible upgrade.
"%PY%" -c "import triton; import importlib.metadata as m; v=m.version('triton-windows'); assert v.startswith('3.6.'), v" >nul 2>&1
if errorlevel 1 (
  echo       Triton-Windows 3.6 missing or incompatible; installing before SageAttention...
  "%PY%" -m pip install --no-warn-script-location --upgrade "triton-windows>=3.6,<3.7"
  if errorlevel 1 goto :fail
) else (
  echo       Compatible Triton-Windows already installed; skipping download.
)

echo [7/8] Installing SageAttention 2.2.0.post6 for CUDA 13 / Torch 2.10+ ABI...
set "SAGE_WHEEL=https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
REM Do not let SageAttention modify the already matched Torch/CUDA stack.
"%PY%" -c "import sageattention; from sageattention import sageattn_qk_int8_pv_fp16_cuda" >nul 2>&1
if errorlevel 1 (
  echo       SageAttention missing or incomplete; installing exact prebuilt wheel with --no-deps...
  "%PY%" -m pip install --no-warn-script-location --no-deps "%SAGE_WHEEL%"
  if errorlevel 1 goto :fail
) else (
  echo       Compatible SageAttention already installed; skipping download.
)

echo [8/8] Running import / CUDA / SageAttention smoke test...
"%PY%" "%CD%\runtime\validate_install.py"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo RUNTIME INSTALL COMPLETE
echo ============================================================
if /i "%MINIMAX_H3_GUI_INSTALL%"=="1" (
  echo FrameVision automatic install mode: runtime complete.
  echo Model and LoRA downloads will be handled automatically by FrameVision.
  exit /b 0
)

echo.
echo Starting model downloader...
echo.
call "%CD%\presets\download_models.bat"
set "DLERR=%ERRORLEVEL%"
if not "%DLERR%"=="0" (
  echo.
  echo Runtime installation completed, but model download returned code %DLERR%.
  echo You can run presets\download_models.bat again at any time.
  pause
  exit /b %DLERR%
)

echo.
echo ============================================================
echo INSTALL AND MODEL DOWNLOAD COMPLETE
echo ============================================================
echo Run start.bat to open MiniMax-H3.
pause
exit /b 0

:fail
echo.
echo INSTALL FAILED. Error code: %ERRORLEVEL%
pause
exit /b 1
