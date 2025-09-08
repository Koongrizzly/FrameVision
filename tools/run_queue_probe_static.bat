@echo off
cd /d "%~dp0"
echo Running static queue probe...
python "%~dp0\queue_probe.py"
pause
