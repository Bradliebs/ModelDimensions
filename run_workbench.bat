@echo off
REM Launch the ModelDimensions Streamlit workbench UI.
REM Works regardless of the current directory: it uses this file's own folder.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not find .venv\Scripts\python.exe in "%~dp0".
    echo Create the virtual environment first, then re-run this script.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app\workbench.py
