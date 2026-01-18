@echo off
title Saudi Stock AI Analyzer
echo ============================================
echo    Saudi Stock AI Analyzer - Launcher
echo ============================================
echo.

cd /d "%~dp0"

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "python app.py"

echo.
echo Starting Frontend (React)...
cd frontend
start "Frontend" cmd /k "npm start"

echo.
echo ============================================
echo Both servers are starting...
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo ============================================
echo.
echo You can close this window.
pause
