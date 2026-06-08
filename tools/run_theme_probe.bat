
@echo off
setlocal
echo Running theme probe...
".\.venv\Scripts\activate" >nul 2>&1 && python -m tools.theme_probe
pause
