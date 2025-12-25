@echo off
setlocal
REM Open venv and open the TXT→IMG pane file for editing

REM jump to project root (this BAT lives in tools\)
cd /d "%~dp0\.."

if not exist ".venv\Scripts\activate.bat" (
  echo [info] No .venv detected. Creating one so editors & linters can resolve imports...
  py -3 -m venv .venv || python -m venv .venv
)

call ".venv\Scripts\activate.bat"
echo [info] VENV: %VIRTUAL_ENV%

echo [info] Opening helpers\txttoimg.py in your default editor...
start "" "helpers\txttoimg.py"

echo.
echo [info] You can now edit the pane. Close this window or press any key.
pause >nul
