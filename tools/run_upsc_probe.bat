@echo off
cd /d "%~dp0"
echo Running upscaler probe...
python "%~dp0\upsc_probe.py"
pause
