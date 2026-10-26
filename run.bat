@echo off
title AI Fake News Detector
cd /d "%~dp0"

echo ===================================================
echo       AI Fake News Detector and Verifier
echo ===================================================
echo.

:: Check for virtual environment if present
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

echo Starting server...
echo URL: http://127.0.0.1:5000
echo.

:: Automatically open default browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Start the Flask application
python app.py

if errorlevel 1 (
    echo.
    echo Server stopped with an error.
)
pause
