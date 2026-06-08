@echo off
setlocal

rem ============================================================
rem FrameVision Startup Profile Check
rem ============================================================
rem This script enables startup profiling for this launch only.
rem Profile logs will be written to:
rem   logs\startup_profile_*.txt
rem   logs\startup_profile_latest.json
rem ============================================================

echo.
echo [FrameVision] Startup profile check
echo [FrameVision] This run will create startup profile logs in the logs folder.
echo.

rem Resolve FrameVision root as the parent folder of this scripts folder.
for %%I in ("%~dp0..") do set "FV_ROOT=%%~fI"

cd /d "%FV_ROOT%" || (
    echo [FrameVision] ERROR: Could not change to FrameVision root:
    echo   "%FV_ROOT%"
    echo.
    pause
    exit /b 1
)

echo [FrameVision] Root: %CD%
echo.

set "FRAMEVISION_STARTUP_PROFILE=1"

if exist "%CD%\start.bat" (
    echo [FrameVision] Starting normal FrameVision launcher path with profiling enabled...
    echo [FrameVision] Profile output:
    echo   logs\startup_profile_*.txt
    echo   logs\startup_profile_latest.json
    echo.
    call "%CD%\start.bat"
    echo.
    echo [FrameVision] start.bat finished or closed.
    echo [FrameVision] Check the logs folder for startup_profile files.
    echo.
    pause
    exit /b 0
)

echo [FrameVision] WARNING: start.bat was not found next to this FrameVision root.
echo [FrameVision] Expected:
echo   "%CD%\start.bat"
echo.

echo [FrameVision] Trying fallback direct runner path...
if exist "%CD%\.venv\Scripts\python.exe" if exist "%CD%\framevision_run.py" (
    echo [FrameVision] Fallback: .venv\Scripts\python.exe framevision_run.py
    echo.
    "%CD%\.venv\Scripts\python.exe" "%CD%\framevision_run.py"
    echo.
    echo [FrameVision] Fallback run finished or closed.
    echo [FrameVision] Check the logs folder for startup_profile files.
    echo.
    pause
    exit /b 0
)

echo [FrameVision] ERROR: Could not run startup profile check.
echo [FrameVision] Missing start.bat, and fallback files were not found:
echo   "%CD%\.venv\Scripts\python.exe"
echo   "%CD%\framevision_run.py"
echo.
pause
exit /b 1
