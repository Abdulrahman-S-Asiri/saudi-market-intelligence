@echo off
title Saudi Stock AI Analyzer v3.0 - Advanced LSTM Edition
echo ============================================================
echo    Saudi Stock AI Analyzer v3.0
echo    Advanced LSTM Edition (BiLSTM + Multi-Head Attention)
echo ============================================================
echo.

cd /d "%~dp0"

echo [*] Starting Backend Server (FastAPI)...
start "Backend Server - Advanced LSTM" cmd /k "cd backend && python app.py"

echo.
echo [*] Starting Frontend (React)...
start "Frontend - TASI AI Analyzer" cmd /k "cd frontend && npm start"

echo.
echo ============================================================
echo Both servers are starting...
echo.
echo   Backend API:  http://localhost:8000
echo   Frontend UI:  http://localhost:3000
echo   API Docs:     http://localhost:8000/docs
echo.
echo   Model: BiLSTM + Multi-Head Attention
echo   Features: 35+ technical indicators
echo ============================================================
echo.
echo You can close this window.
pause
