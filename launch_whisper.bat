@echo off
:: launch_whisper.bat
:: Double-click to start Whisper STT only (no Claude Code).
:: Kills any existing instance first.

powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launch_whisper.ps1"
