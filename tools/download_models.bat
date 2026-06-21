@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "APP_ROOT=%CD%\.."
pushd "%APP_ROOT%"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [setup] Creating virtual environment...
  py -3 -m venv ".venv" || python -m venv ".venv"
  if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
)

echo [run] Downloading describe  + upscaler packs...
python "scripts\download_externals.py" --all
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo [run] downloader exited with errorlevel %ERR%
) else (
  echo [run] done.
)
echo.
echo Overrides:
echo   - Env:   set SWINIR_ZIP_URL=https://...zip  (same for LAPSRN_/REALESRGAN_/WAIFU2X_) 
echo   - CLI:   python scripts\download_externals.py --swinir-only --swinir-url https://...zip
echo   - File:  echo https://...zip > .urls\swinir.txt  (or lapsrn.txt)
echo Offline:
echo   - Drop the zip into .dl_cache\ or manual_models\ and rerun
echo.
pause
popd
REM >>> FRAMEVISION_QWEN_BEGIN
echo   Q) Qwen TXT->IMG (GGUF) optional download
REM <<< FRAMEVISION_QWEN_END

REM >>> FRAMEVISION_QWEN_BEGIN
:QWEN_TOOL_PROMPT
echo.
echo Select VRAM tier for Qwen (TXT->IMG):
echo   1^) 6 -8 GB  (Q3_K_M)
echo   2^) 12-24 GB (Q4_K_M)
set /p TIER=Enter choice 1-2 (or leave blank to cancel): 
if "%TIER%"=="1" goto QWEN_TOOL_Q3
if "%TIER%"=="2" goto QWEN_TOOL_Q4
if "%TIER%"=="" goto END
echo Invalid choice.
pts\python.exe" scripts\download_externals.py --component qwen --tier 12 --dest ".\models\qwen_image"
goto END
REM <<< FRAMEVISION_QWEN_END
