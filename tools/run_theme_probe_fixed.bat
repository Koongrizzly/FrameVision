
@echo off
setlocal
echo Running theme probe (fixed)...
REM Move to project root (parent of tools)
cd /d "%~dp0.."

set VENV_ACT=.venv\Scripts\activate.bat
if exist "%VENV_ACT%" (
  call "%VENV_ACT%"
) else (
  echo [probe] No venv at "%CD%\%VENV_ACT%". Using system Python...
)

python -X utf8 -m tools.theme_probe
echo [probe] exit code: %errorlevel%
echo.
pause
endlocal
