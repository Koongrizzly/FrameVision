@echo off
cd /d "%~dp0"
echo Running queue probe...
python "%~dp0\queue_probe.py"
pause
