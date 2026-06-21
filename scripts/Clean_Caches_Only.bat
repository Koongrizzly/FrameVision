
@echo off
setlocal
REM === Clean_Caches_Only.bat ===
for /r %%D in (__pycache__) do if exist "%%D" rmdir /s /q "%%D"
for /r %%F in (*.pyc) do del /f /q "%%F"
echo Caches removed.
pause
endlocal
