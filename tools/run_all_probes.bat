@echo off
REM FrameVision: run all probes (modules + analyzer + theme)
REM Place this file into your 'tools' folder and double-click.

setlocal ENABLEEXTENSIONS
pushd "%~dp0"

echo === Running modules_probe ===
if exist "modules_probe.bat" (
  call modules_probe.bat
) else (
  echo (modules_probe.bat not found — skipping)
)

echo === Running run_analyzer ===
if exist "run_analyzer.bat" (
  call run_analyzer.bat
) else (
  echo (run_analyzer.bat not found — skipping)
)

echo === Running theme probe ===
if exist "run_theme_probe_fixed.bat" (
  call run_theme_probe_fixed.bat
) else if exist "run_theme_probe.bat" (
  call run_theme_probe.bat
) else if exist "theme_probe.bat" (
  call theme_probe.bat
) else (
  echo (no theme probe found — skipping)
)

echo === All probes completed ===
pause
popd