@echo off
REM Start Bascol Capture System with WebSocket Server

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================
echo   Bascol Capture System - Starting...
echo ============================================
echo.

REM Start WebSocket Server
echo [1/2] Starting WebSocket Server (ws://127.0.0.1:2020)...
start "WebSocket Server" cmd /k "cd /d "%~dp0" && python311 servers/websocket_server.py"
timeout /t 2 /nobreak

REM Start Image Server
echo [2/2] Starting Image Server (http://127.0.0.1:3000)...
start "Image Server" cmd /k "cd /d "%~dp0" && python311 servers/image_server.py"
timeout /t 2 /nobreak

REM Start Demo
echo [3/3] Starting Demo (listening for RFID tags)...
echo.
echo ============================================
echo   System Ready!
echo ============================================
echo.
echo   WebSocket: ws://127.0.0.1:2020
echo   Images: http://127.0.0.1:3000/images/
echo.
echo   Use Hoppscotch or similar WebSocket client to connect
echo   to: ws://127.0.0.1:2020
echo.
echo ============================================
echo.

python311 demo/main.py

pause
