@echo off
REM Launch the local Bank Management Workspace UI.
REM Works regardless of the current directory: it uses this file's own folder.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not find .venv\Scripts\python.exe in "%~dp0".
    echo Create the virtual environment first, then re-run this script.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/status', timeout=2).read()" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Bank Management Workspace is already running.
    echo Open http://127.0.0.1:8765
    start "" "http://127.0.0.1:8765"
    exit /b 0
)

".venv\Scripts\python.exe" app\bank_workspace.py --host 127.0.0.1 --port 8765