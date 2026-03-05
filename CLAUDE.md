# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows system tray push-to-talk speech-to-text app. Hold a hotkey (default F2), speak, release — transcribed text is pasted at cursor. Powered by faster-whisper running locally (no cloud/API). Single-file Python app (`whisper_type.py`).

## Setup & Running

```bash
# One-time setup (checks Python, installs deps)
# Double-click setup.bat, or:
pip install -r requirements.txt

# Launch (kills any existing instance, starts hidden in system tray)
# Double-click launch_whisper.bat, or:
python whisper_type.py
```

No tests or linter are configured.

## Architecture

Everything lives in `whisper_type.py` (~700 lines):

- **Config constants** (top of file): `HOTKEY`, `WHISPER_MODEL`, `SAMPLE_RATE`, `APPEND_ENTER`, `BUBBLE_DURATION`, `SMOOTHING`
- **`Bubble` class**: Tkinter-based transparent overlay at bottom-center of screen. Runs on its own thread. Shows recording waveform (red), transcribing state (yellow), and result text (green). Has fade animations and live FFT-driven waveform bars.
- **`WhisperTray` class**: Main application. Manages:
  - Model loading with CUDA auto-detection and CPU fallback
  - Audio capture via `sounddevice.InputStream` → buffer → WAV → faster-whisper transcribe
  - Streaming transcription (segments update bubble progressively)
  - Hotkey registration via `keyboard` library (supports single keys and chord combos like `ctrl+alt+r`)
  - System tray icon/menu via `pystray` (hotkey picker, GPU/CPU toggle, chime toggle)
  - Text pasting via `pyperclip` + `pyautogui`

## Key Dependencies

`faster-whisper` (with `ctranslate2`), `sounddevice`, `keyboard`, `pyautogui`, `pyperclip`, `pystray`, `Pillow`, `numpy`. Optional CUDA: `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`.

## Platform

Windows-only. Uses Windows-specific features: `.bat`/`.ps1` launchers, `Win32_Process` WMI for duplicate instance detection, `Segoe UI` font, `pyautogui` Ctrl+V paste.
