@echo off
for /r %%d in (__pycache__) do if exist "%%d" rd /s /q "%%d"
echo Caches cleared.
