@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.ps1 first. See README.md for instructions.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py --demo %*
if errorlevel 1 pause
