
@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "FRAMEVISION_ROOT=%~dp0..\.."
for %%I in ("%FRAMEVISION_ROOT%") do set "FRAMEVISION_ROOT=%%~fI"

set "PYTHON_EXE=%FRAMEVISION_ROOT%\environments\ltx25_convrot\Scripts\python.exe"
set "SCRIPT=%~dp0download_ltx25_comfy_backend.py"

echo ================================================================
echo FrameVision LTX 2.5 - Current isolated Comfy backend downloader
echo ================================================================
echo.
echo Target:
echo   %FRAMEVISION_ROOT%\models\ltx_2_5_convrot\ComfyUI
echo.
echo This does NOT modify:
echo   %FRAMEVISION_ROOT%\vendor
echo   %FRAMEVISION_ROOT%\models\ltx_2_5
echo.
echo Existing Torch in ltx25_convrot is preserved.
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERROR] ConvRot environment not found:
    echo         %PYTHON_EXE%
    echo.
    echo Install W4A8 or INT4 first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [OK] Finished successfully.
) else (
    echo [ERROR] Installer exited with code %RC%.
)

if /I not "%~1"=="--no-pause" pause
exit /b %RC%
