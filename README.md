# Whisper STT -- Push-to-Talk Speech-to-Text for Windows

A system tray app that transcribes your voice and pastes the text wherever your cursor is. Hold a hotkey, speak, release -- done.

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running locally. No cloud, no API keys, no subscription.

## Quick Start

### Prerequisites
- **Python 3.10+** -- [python.org](https://python.org) (check "Add Python to PATH" during install)
- **Windows 10/11**

### Setup (one time)
1. Double-click **`setup.bat`** -- checks Python and installs dependencies

### Usage
1. Double-click **`launch_whisper.bat`**
2. A green dot appears in your system tray -- the app is ready
3. **Hold Alt+Q** -- speak -- **release Alt+Q**
4. Text is pasted at your cursor

First launch downloads the Whisper model (~150MB). Needs internet once.

---

> ## **Launch on Startup (Optional)**
>
> To have Whisper STT start automatically every time you log in:
>
> **Step 1 -- Open the Windows Startup folder:**
> - Press **Win + R** on your keyboard (opens the Run dialog)
> - Type `shell:startup` and press **Enter**
> - A File Explorer window will open -- this is your Startup folder
>
> **Step 2 -- Create a shortcut:**
> - Navigate to where you saved this project (e.g., your SpeechToText folder)
> - **Right-click** `launch_whisper.bat` -- **Copy**
> - Go back to the Startup folder you opened in Step 1
> - **Right-click** in the empty space -- **Paste shortcut**
>
> **Step 3 -- Done!**
> Next time you restart or log in, Whisper STT will start automatically in the system tray.
>
> *To stop it from auto-starting, just delete the shortcut from the Startup folder.*

---

## Features

- **Live waveform visualizer** -- real-time FFT-driven bars show your audio input while recording
- **System tray controls** -- right-click the tray icon to:
  - Change hotkey (F1-F12, Ctrl+combos, Alt+combos)
  - Toggle GPU/CPU inference
  - Toggle sound chimes on record start/finish
- **GPU acceleration** -- auto-detects CUDA and validates with a quick test transcription at startup. Uses your GPU if it passes, falls back to CPU if not. Toggleable from the tray menu
- **Streaming transcription** -- text appears progressively as Whisper decodes segments
- **Overlay bubble** -- floating indicator at bottom-center of screen:
  - Red waveform = recording
  - Yellow waveform = transcribing
  - Green waveform + white text = result (can be disabled)
- **Smooth animations** -- fade in/out transitions, smoothed FFT bars
- **Duplicate instance protection** -- launching again kills the old instance first

## Files

| File | Purpose |
|---|---|
| `whisper_type.py` | Main app -- all logic in one file |
| `launch_whisper.bat` | Start STT (daily driver) |
| `setup.bat` | One-time setup (installs deps) |
| `requirements.txt` | Python dependencies |

## Configuration

Edit the top of `whisper_type.py`:

| Setting | Default | Description |
|---|---|---|
| `HOTKEY` | `alt+q` | Hold to record, release to transcribe |
| `WHISPER_MODEL` | `base.en` | Model size: `tiny.en`, `base.en`, `small.en`, `medium.en` |
| `APPEND_ENTER` | `False` | Press Enter after pasting (auto-submit) |
| `BUBBLE_DURATION` | `0` | Seconds to show result text (0 = no result bubble) |
| `SMOOTHING` | `0.45` | FFT bar smoothing (0 = raw, 1 = frozen) |

## GPU Acceleration

If you have an NVIDIA GPU, the app auto-detects CUDA and runs a quick test transcription at startup to verify it works. If the test passes, it uses `float16` inference (significantly faster). If the test fails or times out, it falls back to CPU automatically.

You can also toggle between GPU and CPU from the tray menu at any time.

For CUDA support, `ctranslate2` (installed with `faster-whisper`) needs the CUDA runtime libraries. If not detected, install:
```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Sharing with Coworkers

1. Zip this entire folder
2. Send it
3. They double-click `setup.bat` (needs Python installed)
4. Run `launch_whisper.bat` -- done
