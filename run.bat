@echo off
title AI Fake News Detector
cd /d "%~dp0"

echo ===================================================
echo       AI Fake News Detector and Verifier
echo ===================================================
echo.

:: Check if server is already running on port 5000
netstat -ano | findstr :5000 | findstr LISTENING >nul
if %errorlevel% == 0 (
    echo Server is already active and running!
    echo Opening application in your web browser: http://127.0.0.1:5000
    start http://127.0.0.1:5000
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)

:: Check for virtual environment if present
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment [venv]
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment [.venv]
    call .venv\Scripts\activate.bat
)

echo Starting server...
echo URL: http://127.0.0.1:5000
echo.

:: Automatically open default browser after brief delay
start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://127.0.0.1:5000"

:: Start the Flask application
python app.py

if errorlevel 1 (
    echo.
    echo Server stopped with an error.
)
pause
