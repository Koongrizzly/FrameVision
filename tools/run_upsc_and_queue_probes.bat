@echo off
cd /d "%~dp0"
echo === Upscaler probe ===
python "%~dp0\upsc_probe.py"
echo.
echo === Queue probe ===
python "%~dp0\queue_probe.py"
pause
