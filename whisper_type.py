#!/usr/bin/env python3
"""
whisper_type.py - System tray push-to-talk STT.

Runs as a background process with a system tray icon.
Hold F2 to record, release to transcribe, result is pasted at your cursor.

No dedicated terminal pane needed -- just launch and forget.

Dependencies:
    pip install faster-whisper sounddevice numpy pyautogui keyboard pyperclip pystray Pillow
    (optional) pip install nvidia-cublas-cu12 nvidia-cudnn-cu12  -- for GPU acceleration
"""

import sys
import time
import os
import wave
import threading
import tempfile
import tkinter as tk
import numpy as np

# ---------------------------------------------
#  CONFIG
# ---------------------------------------------
HOTKEY          = "alt+q"            # Hold to record, release to transcribe + paste
WHISPER_MODEL   = "base.en"       # tiny.en | base.en | small.en | medium.en
SAMPLE_RATE     = 16000
APPEND_ENTER    = False           # Auto-submit with Enter after pasting
BUBBLE_DURATION = 0             # Seconds to show the result bubble (0 = no result bubble)
SMOOTHING       = 0.45            # FFT bar smoothing (0 = no smoothing, 1 = frozen)
PRIVACY_MIC     = True            # Only open mic while recording (closes between recordings)
# ---------------------------------------------

NUM_BARS = 10
FFT_CHUNK = 2048  # samples for FFT (~128ms at 16kHz)
CHIME_RATE = 44100  # sample rate for chime playback


def _generate_chime(freq, duration=0.12, volume=0.3, fade=0.03):
    """Generate a short sine-wave chime as a numpy array."""
    t = np.linspace(0, duration, int(CHIME_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    # Apply fade-in and fade-out to avoid clicks
    fade_samples = int(CHIME_RATE * fade)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return tone.astype(np.float32)


# Pre-generate chimes: higher pitch = start, two-tone rising = done
CHIME_START = _generate_chime(880, duration=0.10)   # A5 -- short blip
_done_lo = _generate_chime(880, duration=0.08)
_done_hi = _generate_chime(1175, duration=0.12)     # D6
# Small gap then second note
_gap = np.zeros(int(CHIME_RATE * 0.03), dtype=np.float32)
CHIME_DONE = np.concatenate([_done_lo, _gap, _done_hi])

# Available hotkey choices for the tray menu
# Single keys and chord combos both supported
HOTKEY_OPTIONS = [
    # Function keys
    "f1", "f2", "f3", "f4", "f5", "f6",
    "f7", "f8", "f9", "f10", "f11", "f12",
    # Ctrl combos
    "ctrl+`", "ctrl+\\",
    "ctrl+alt+r", "ctrl+alt+s", "ctrl+alt+w",
    # Alt combos
    "alt+`", "alt+\\", "alt+q"
]


def check_deps():
    missing = []
    for mod in ["faster_whisper", "sounddevice", "numpy", "pyautogui",
                "keyboard", "pyperclip", "pystray", "PIL"]:
        try:
            __import__(mod)
        except ImportError:
            name = mod.replace("_", "-")
            if mod == "PIL":
                name = "Pillow"
            missing.append(name)
    if missing:
        print(f"[ERROR] Missing deps: {', '.join(missing)}")
        print(f"[FIX  ] pip install {' '.join(missing)}")
        sys.exit(1)


def detect_cuda():
    """Check if CUDA is available for faster-whisper."""
    try:
        import ctranslate2
        # If this call succeeds, CUDA is available. It returns compute types like float16, int8, etc.
        types = ctranslate2.get_supported_compute_types("cuda")
        return len(types) > 0
    except Exception:
        return False


def make_icon_image(color):
    """Create a simple colored circle icon for the system tray."""
    from PIL import Image, ImageDraw
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


def draw_waveform(size=96, color="#F44336", bar_heights=None):
    """Draw waveform equalizer bars and return a Pillow RGBA Image."""
    from PIL import Image, ImageDraw

    if bar_heights is None:
        bar_heights = [0.3, 0.6, 0.9, 0.6, 0.3]

    num_bars = len(bar_heights)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    padding = size * 0.15
    usable_w = size - 2 * padding
    gap_ratio = 0.4
    bar_w = usable_w / (num_bars + (num_bars - 1) * gap_ratio)
    gap = bar_w * gap_ratio
    max_h = size * 0.70
    cy = size / 2
    radius = bar_w / 2

    for i, h in enumerate(bar_heights):
        x = padding + i * (bar_w + gap)
        bar_h = max(bar_w, max_h * h)
        top = cy - bar_h / 2
        bot = cy + bar_h / 2
        draw.rounded_rectangle(
            [x, top, x + bar_w, bot],
            radius=radius,
            fill=color,
        )

    return img


# ---------------------------------------------
#  Overlay Bubble (tkinter, own thread)
# ---------------------------------------------
class Bubble:
    """A small popup at bottom-center of the screen."""

    COLORS = {
        "recording":    {"bg": "#F44336", "fg": "#FFFFFF"},
        "transcribing": {"bg": "#FFC107", "fg": "#000000"},
        "result":       {"bg": "#4CAF50", "fg": "#FFFFFF"},
    }

    FADE_STEPS  = 12
    FADE_MS     = 16
    MAX_ALPHA   = 0.92
    TRANSPARENT = "#010101"
    MIC_SIZE    = 120

    def __init__(self, fft_callback=None):
        self._root = None
        self._label = None
        self._mic_label = None
        self._mic_images = {}
        self._anim_job = None
        self._fft_callback = fft_callback
        self._ready = threading.Event()
        self._hide_timer = None
        self._fade_job = None
        self._alpha = 0.0
        self._current_style = None
        # Smoothed bar heights for interpolation
        self._smooth_bars = [0.15] * NUM_BARS

        t = threading.Thread(target=self._tk_main, daemon=True)
        t.start()
        self._ready.wait()

    def _build_waveform_photo(self, color_hex, bar_heights=None):
        from PIL import ImageTk
        pil_img = draw_waveform(size=self.MIC_SIZE, color=color_hex, bar_heights=bar_heights)
        photo = ImageTk.PhotoImage(pil_img)
        return photo

    def _tk_main(self):
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.0)
        self._root.attributes("-transparentcolor", self.TRANSPARENT)
        self._root.withdraw()

        self._frame = tk.Frame(self._root, bg=self.TRANSPARENT)
        self._frame.pack()

        self._mic_label = tk.Label(
            self._frame,
            bg=self.TRANSPARENT,
            borderwidth=0,
        )

        self._label = tk.Label(
            self._frame,
            text="",
            font=("Segoe UI", 18, "bold"),
            padx=16,
            pady=10,
            wraplength=500,
        )

        self._mic_images["result"] = self._build_waveform_photo("#4CAF50")

        self._screen_w = self._root.winfo_screenwidth()
        self._screen_h = self._root.winfo_screenheight()

        self._ready.set()
        self._root.mainloop()

    def _position_bottom_center(self):
        self._root.update_idletasks()
        ww = self._root.winfo_reqwidth()
        wh = self._root.winfo_reqheight()
        x = (self._screen_w - ww) // 2
        y = self._screen_h - wh - 80
        self._root.geometry(f"{ww}x{wh}+{x}+{y}")

    def _cancel_animations(self):
        if self._fade_job is not None:
            self._root.after_cancel(self._fade_job)
            self._fade_job = None
        if self._hide_timer is not None:
            self._root.after_cancel(self._hide_timer)
            self._hide_timer = None
        if self._anim_job is not None:
            self._root.after_cancel(self._anim_job)
            self._anim_job = None

    def _start_live_fft(self, color_hex):
        """Poll live FFT data, smooth it, and redraw the waveform."""
        def _tick():
            if self._fft_callback:
                raw = self._fft_callback()
                # Exponential smoothing for fluid bar motion
                for i in range(NUM_BARS):
                    self._smooth_bars[i] += (raw[i] - self._smooth_bars[i]) * (1.0 - SMOOTHING)
                photo = self._build_waveform_photo(color_hex, bar_heights=self._smooth_bars)
                self._mic_label.config(image=photo)
                self._mic_label._live_photo = photo
            self._anim_job = self._root.after(50, _tick)  # ~20fps

        self._smooth_bars = [0.15] * NUM_BARS
        _tick()

    def _fade_to(self, target, on_done=None):
        step = (target - self._alpha) / self.FADE_STEPS
        remaining = self.FADE_STEPS

        def _tick():
            nonlocal remaining
            remaining -= 1
            if remaining <= 0:
                self._alpha = target
            else:
                self._alpha += step
            self._root.attributes("-alpha", max(0.0, min(1.0, self._alpha)))

            if remaining > 0:
                self._fade_job = self._root.after(self.FADE_MS, _tick)
            else:
                self._fade_job = None
                if on_done:
                    on_done()

        self._fade_job = self._root.after(self.FADE_MS, _tick)

    def show(self, text, style="recording", duration=None):
        def _update():
            self._cancel_animations()
            self._current_style = style

            self._mic_label.pack_forget()
            self._label.pack_forget()

            if style == "result":
                self._root.config(bg=self.TRANSPARENT)
                self._frame.config(bg=self.TRANSPARENT)
                self._root.attributes("-transparentcolor", self.TRANSPARENT)

                self._mic_label.config(image=self._mic_images["result"], bg=self.TRANSPARENT)
                self._mic_label.pack(side="left", padx=(8, 0), pady=4)

                self._label.config(text=text, bg=self.TRANSPARENT, fg="#FFFFFF")
                self._label.pack(side="left", padx=(4, 16), pady=4)
            else:
                self._root.config(bg=self.TRANSPARENT)
                self._frame.config(bg=self.TRANSPARENT)
                self._root.attributes("-transparentcolor", self.TRANSPARENT)

                color = self.COLORS[style]["bg"]
                self._mic_label.config(bg=self.TRANSPARENT)
                self._mic_label.pack()
                self._start_live_fft(color)

            self._position_bottom_center()
            self._root.deiconify()
            self._root.lift()

            self._fade_to(self.MAX_ALPHA)

            if duration:
                self._hide_timer = self._root.after(
                    int(duration * 1000), self._do_hide
                )

        self._root.after(0, _update)

    def hide(self):
        self._root.after(0, self._do_hide)

    def _do_hide(self):
        self._hide_timer = None
        if self._fade_job is not None:
            self._root.after_cancel(self._fade_job)
            self._fade_job = None
        self._fade_to(0.0, on_done=self._root.withdraw)

    def destroy(self):
        self._root.after(0, self._root.destroy)


# ---------------------------------------------
#  Main app
# ---------------------------------------------
class WhisperTray:
    COLOR_READY       = (76, 175, 80, 255)
    COLOR_RECORDING   = (244, 67, 54, 255)
    COLOR_TRANSCRIBING = (255, 193, 7, 255)

    def __init__(self):
        self.model = None
        self.audio_buffer = []
        self.is_recording = False
        self.start_time = None
        self.stream = None
        self.tray = None
        self.bubble = None
        self._stop_event = threading.Event()
        self._viz_buffer = np.zeros(FFT_CHUNK, dtype=np.float32)
        # Device management
        self._cuda_available = detect_cuda()
        self._use_cuda = self._cuda_available  # default to GPU if available
        self._hotkey = HOTKEY
        self._chime_enabled = False
        self._privacy_mic = PRIVACY_MIC
        # Streaming transcription state
        self._stream_text = ""
        self._stream_lock = threading.Lock()

    @property
    def _device(self):
        return "cuda" if self._use_cuda else "cpu"

    @property
    def _compute_type(self):
        return "float16" if self._use_cuda else "int8"

    def _test_cuda_inference(self, timeout=4):
        """Spawn a subprocess to test CUDA transcription. Returns True if it works."""
        import subprocess
        test_script = (
            "import sys, tempfile, wave, numpy as np\n"
            "from faster_whisper import WhisperModel\n"
            f"model = WhisperModel('{WHISPER_MODEL}', device='cuda', compute_type='float16')\n"
            "path = tempfile.mktemp(suffix='.wav')\n"
            "pcm = np.zeros(16000, dtype=np.int16)\n"  # 1s silence
            "with wave.open(path, 'w') as wf:\n"
            "    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)\n"
            "    wf.writeframes(pcm.tobytes())\n"
            "segs, _ = model.transcribe(path, beam_size=1)\n"
            "list(segs)\n"  # force iteration
            "import os; os.unlink(path)\n"
            "print('OK')\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0 and "OK" in result.stdout
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def load_model(self):
        from faster_whisper import WhisperModel

        if self._use_cuda:
            print(f"[INIT ] CUDA detected, testing GPU inference...")
            if self._test_cuda_inference():
                print(f"[GPU  ] CUDA test passed.")
            else:
                print(f"[WARN ] CUDA test failed or timed out, using CPU instead.")
                self._use_cuda = False

        device = self._device
        compute = self._compute_type
        print(f"[INIT ] Loading Whisper '{WHISPER_MODEL}' on {device} ({compute})...")
        try:
            self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
        except Exception as e:
            if device == "cuda":
                print(f"[WARN ] CUDA failed ({e}), falling back to CPU...")
                self._use_cuda = False
                self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            else:
                raise
        print(f"[READY] Model loaded on {self._device}. Hold {self._hotkey.upper()} to record.")

    def reload_model(self):
        """Reload the model on the current device setting."""
        from faster_whisper import WhisperModel
        device = self._device
        compute = self._compute_type
        print(f"[INIT ] Reloading Whisper on {device} ({compute})...")
        try:
            self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
            print(f"[READY] Model reloaded on {device}.")
        except Exception as e:
            if device == "cuda":
                print(f"[WARN ] CUDA failed ({e}), falling back to CPU...")
                self._use_cuda = False
                self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                print(f"[READY] Model reloaded on CPU.")
            else:
                raise
        self._rebuild_tray_menu()

    def play_chime(self, chime_data):
        """Play a chime on a background thread (non-blocking)."""
        if not self._chime_enabled:
            return
        def _play():
            import sounddevice as sd
            sd.play(chime_data, samplerate=CHIME_RATE, blocking=True)
        threading.Thread(target=_play, daemon=True).start()

    def set_icon(self, color):
        if self.tray:
            self.tray.icon = make_icon_image(color)

    def _open_audio_stream(self):
        """Open the microphone input stream."""
        if self.stream is not None:
            return
        self.stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
            blocksize=1024,
        )
        self.stream.start()

    def _close_audio_stream(self):
        """Close the microphone input stream."""
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0]
        if self.is_recording:
            self.audio_buffer.append(mono.copy())
            n = len(mono)
            self._viz_buffer = np.roll(self._viz_buffer, -n)
            self._viz_buffer[-n:] = mono

    def get_fft_bars(self):
        window = np.hanning(FFT_CHUNK)
        spectrum = np.abs(np.fft.rfft(self._viz_buffer * window))
        freqs = np.fft.rfftfreq(FFT_CHUNK, 1.0 / SAMPLE_RATE)
        edges = np.logspace(np.log10(60), np.log10(7500), NUM_BARS + 1)
        bars = []
        for i in range(NUM_BARS):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if mask.any():
                bars.append(float(np.mean(spectrum[mask])))
            else:
                bars.append(0.0)
        peak = max(max(bars), 1e-6)
        bars = [min(b / peak, 1.0) * 0.85 + 0.15 for b in bars]
        return bars

    def write_wav(self, path, audio_np):
        pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    def transcribe(self, audio_np):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self.write_wav(tmp_path, audio_np)
            segments, _ = self.model.transcribe(tmp_path, beam_size=5)
            text = " ".join(s.text.strip() for s in segments).strip()
        finally:
            os.unlink(tmp_path)
        return text

    def transcribe_streaming(self, audio_np):
        """Transcribe and update the bubble progressively as segments arrive."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self.write_wav(tmp_path, audio_np)
            segments, _ = self.model.transcribe(tmp_path, beam_size=5)
            parts = []
            for seg in segments:
                parts.append(seg.text.strip())
                text_so_far = " ".join(parts)
                with self._stream_lock:
                    self._stream_text = text_so_far
                if BUBBLE_DURATION > 0:
                    self.bubble.show(text_so_far, style="result", duration=None)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return self._stream_text

    def paste_text(self, text):
        import pyperclip
        import pyautogui
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        if APPEND_ENTER:
            time.sleep(0.05)
            pyautogui.press("enter")

    def on_key_down(self):
        if self.is_recording:
            return
        if self._privacy_mic:
            self._open_audio_stream()
        self.is_recording = True
        self.audio_buffer.clear()
        self._viz_buffer[:] = 0
        self._stream_text = ""
        self.start_time = time.time()
        self.set_icon(self.COLOR_RECORDING)
        self.bubble.show("Recording...", style="recording")
        self.play_chime(CHIME_START)

    def on_key_up(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self._privacy_mic:
            self._close_audio_stream()
        elapsed = time.time() - self.start_time

        if elapsed < 0.3 or not self.audio_buffer:
            self.set_icon(self.COLOR_READY)
            self.bubble.hide()
            return

        self.set_icon(self.COLOR_TRANSCRIBING)
        self.bubble.show("Transcribing...", style="transcribing")
        audio_np = np.concatenate(self.audio_buffer)
        peak = np.max(np.abs(audio_np))
        rms = np.sqrt(np.mean(audio_np ** 2))
        print(f"[AUDIO] {elapsed:.1f}s, {len(audio_np)} samples, peak={peak:.4f}, rms={rms:.4f}")

        def _do_transcribe():
            text = self.transcribe_streaming(audio_np)
            if text:
                print(f"[TEXT ] {text}")
                self.play_chime(CHIME_DONE)
                if BUBBLE_DURATION > 0:
                    self.bubble.show(text, style="result", duration=BUBBLE_DURATION)
                else:
                    self.bubble.hide()
                self.paste_text(text)
            else:
                print("[EMPTY] Nothing transcribed.")
                self.bubble.show("(nothing heard)", style="transcribing", duration=1.5)
            self.set_icon(self.COLOR_READY)

        threading.Thread(target=_do_transcribe, daemon=True).start()

    # -- Hotkey management ---------------------

    def _is_chord(self, key):
        return "+" in key

    def _register_hotkey(self):
        import keyboard
        if self._is_chord(self._hotkey):
            # Chord combo: fire on_key_down when pressed, on_key_up when released
            keyboard.add_hotkey(self._hotkey, self.on_key_down, suppress=False)
            # For release detection on chords, watch the last key in the combo
            parts = self._hotkey.split("+")
            self._chord_trigger_key = parts[-1].strip()
            keyboard.on_release_key(self._chord_trigger_key, lambda e: self._chord_release_check(), suppress=False)
        else:
            # Single key: straightforward press/release
            keyboard.on_press_key(self._hotkey, lambda e: self.on_key_down(), suppress=False)
            keyboard.on_release_key(self._hotkey, lambda e: self.on_key_up(), suppress=False)
            self._chord_trigger_key = None

    def _chord_release_check(self):
        """For chord hotkeys, trigger on_key_up when the trigger key is released."""
        if self.is_recording:
            self.on_key_up()

    def _unregister_hotkey(self):
        import keyboard
        keyboard.unhook_all()

    def change_hotkey(self, new_key):
        self._unregister_hotkey()
        self._hotkey = new_key
        self._register_hotkey()
        print(f"[KEY  ] Hotkey changed to {new_key.upper()}")
        self._rebuild_tray_menu()

    def hotkey_loop(self):
        self._register_hotkey()
        self._stop_event.wait()

    # -- Device toggle -------------------------

    def toggle_device(self):
        if not self._cuda_available:
            print("[WARN ] CUDA not available on this system.")
            return
        self._use_cuda = not self._use_cuda
        threading.Thread(target=self.reload_model, daemon=True).start()

    def _toggle_chime(self):
        self._chime_enabled = not self._chime_enabled
        state = "on" if self._chime_enabled else "off"
        print(f"[CHIME] Sound chimes {state}")

    def _toggle_privacy_mic(self):
        self._privacy_mic = not self._privacy_mic
        if self._privacy_mic:
            self._close_audio_stream()
            print("[MIC  ] Privacy mic mode enabled: stream opens only while recording.")
        else:
            self._open_audio_stream()
            print("[MIC  ] Privacy mic mode disabled: always-on mic stream started.")
        self._rebuild_tray_menu()

    # -- Tray ----------------------------------

    def _rebuild_tray_menu(self):
        import pystray

        device_label = f"Device: {self._device.upper()}"
        if self._cuda_available:
            device_item = pystray.MenuItem(
                device_label + " (click to toggle)",
                lambda icon, item: self.toggle_device(),
            )
        else:
            device_item = pystray.MenuItem(
                device_label + " (CUDA not available)",
                lambda: None,
                enabled=False,
            )

        # Build hotkey submenu with categories
        def _make_hotkey_handler(key):
            return lambda icon, item: self.change_hotkey(key)

        def _make_hotkey_item(key):
            return pystray.MenuItem(
                key.upper(),
                _make_hotkey_handler(key),
                checked=lambda item, k=key: k == self._hotkey,
                radio=True,
            )

        fkeys = [k for k in HOTKEY_OPTIONS if not "+" in k]
        chords = [k for k in HOTKEY_OPTIONS if "+" in k]

        hotkey_items = [_make_hotkey_item(k) for k in fkeys]
        if chords:
            hotkey_items.append(pystray.Menu.SEPARATOR)
            hotkey_items.extend([_make_hotkey_item(k) for k in chords])

        chime_item = pystray.MenuItem(
            "Sound chimes",
            lambda icon, item: self._toggle_chime(),
            checked=lambda item: self._chime_enabled,
        )

        privacy_item = pystray.MenuItem(
            "Privacy mic (open only while recording)",
            lambda icon, item: self._toggle_privacy_mic(),
            checked=lambda item: self._privacy_mic,
        )

        menu = pystray.Menu(
            pystray.MenuItem(f"Hold {self._hotkey.upper()} to record", lambda: None, enabled=False),
            pystray.MenuItem(f"Model: {WHISPER_MODEL}", lambda: None, enabled=False),
            device_item,
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Hotkey", pystray.Menu(*hotkey_items)),
            chime_item,
            privacy_item,
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app),
        )

        if self.tray:
            self.tray.menu = menu
            self.tray.update_menu()

    def quit_app(self, icon, item):
        self._stop_event.set()
        if self.stream:
            self.stream.close()
        if self.bubble:
            self.bubble.destroy()
        icon.stop()

    def run(self):
        import pystray
        import sounddevice as sd

        check_deps()

        if self._cuda_available:
            print(f"[GPU  ] CUDA detected! Using GPU acceleration.")
        else:
            print(f"[CPU  ] No CUDA detected, using CPU.")

        self.load_model()

        # Start overlay bubble with live FFT feed
        self.bubble = Bubble(fft_callback=self.get_fft_bars)

        # Audio stream setup
        self._sd = sd
        default_dev = sd.query_devices(kind='input')
        print(f"[MIC  ] Using: {default_dev['name']} (channels={default_dev['max_input_channels']})")
        self.stream = None
        if self._privacy_mic:
            print(f"[MIC  ] Privacy mic mode: stream opens only while recording.")
        else:
            self._open_audio_stream()
            print(f"[MIC  ] Always-on mic stream started.")

        # Start hotkey listener in background thread
        hotkey_thread = threading.Thread(target=self.hotkey_loop, daemon=True)
        hotkey_thread.start()

        # Build initial tray icon + menu
        self.tray = pystray.Icon(
            "whisper_type",
            make_icon_image(self.COLOR_READY),
            "Whisper STT (Ready)",
        )
        self._rebuild_tray_menu()

        print(f"[TRAY ] Running in system tray. Hold {self._hotkey.upper()} to record.")
        self.tray.run()


def main():
    app = WhisperTray()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[BYE  ] Later! o/")


if __name__ == "__main__":
    main()
