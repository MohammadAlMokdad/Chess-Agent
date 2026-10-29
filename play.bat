@echo off
rem One-click launcher: calibrate on a fresh game, then play it.
rem Double-click this file, or run it from a terminal with extra options:
rem     play.bat --strength 6
rem     play.bat --delay 1 2
cd /d "%~dp0"

echo ------------------------------------------------------------
echo  Chess Agent
echo ------------------------------------------------------------
echo  1. Start a NEW game on your chess site (pieces on home squares)
echo  2. Leave the board fully visible, nothing covering it
echo  3. Do not touch the mouse once it starts
echo.
echo  Press F9 at any time to stop.
echo ------------------------------------------------------------
echo.

python run.py calibrate --retry 400
if errorlevel 1 (
    echo.
    echo Calibration failed. It needs a fresh STARTING position on screen.
    echo If a game is already underway, start a new one and run this again.
    echo.
    pause
    exit /b 1
)

echo.
python run.py play --think 0.4 %*

echo.
pause
