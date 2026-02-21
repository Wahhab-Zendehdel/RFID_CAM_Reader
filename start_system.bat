@echo off
REM Start Bascol Capture System with REST API

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================
echo   Bascol Capture System - Starting...
echo ============================================
echo.

REM Start Data API Server
echo [1/3] Starting Data API Server (http://127.0.0.1:8000)...
start "Data API Server" cmd /k "cd /d "%~dp0" && python311 -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak

REM Start Image Server
echo [2/3] Starting Image Server (http://127.0.0.1:3000)...
start "Image Server" cmd /k "cd /d "%~dp0" && python311 servers/image_server.py"
timeout /t 2 /nobreak

REM Start Demo
echo [3/3] Starting Demo (listening for RFID tags)...
echo.
echo ============================================
echo   System Ready!
echo ============================================
echo.
echo   Data API: http://127.0.0.1:8000
echo   Swagger: http://127.0.0.1:8000/docs
echo   Images: http://127.0.0.1:3000/images/
echo.
echo   Use Swagger UI for CRUD operations:
echo   http://127.0.0.1:8000/docs
echo.
echo ============================================
echo.

python311 demo/main.py

pause
