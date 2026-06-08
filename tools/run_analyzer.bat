
@echo off
setlocal
REM tools\run_analyzer.bat — activates venv and runs analyzer from project root
set ROOT=%~dp0..
pushd "%ROOT%"
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)
python -m tools.fv_analyzer
popd
endlocal
pause
