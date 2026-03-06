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
import json
import ctypes
import urllib.request
import urllib.error
import tkinter as tk
import numpy as np

# ---------------------------------------------
#  CONFIG
# ---------------------------------------------
HOTKEY          = "alt+q"            # Hold to record, release to transcribe + paste
WHISPER_MODEL   = "base.en"       # tiny.en | base.en | small.en | medium.en
SAMPLE_RATE     = 16000
APPEND_ENTER    = False           # Auto-submit with Enter after pasting
BUBBLE_DURATION = 3             # Seconds to show the result bubble (0 = no result bubble)
SMOOTHING       = 0.45            # FFT bar smoothing (0 = no smoothing, 1 = frozen)
LLM_CLEANUP     = True            # Post-process transcription with a local LLM
LLM_MODEL       = "llama3.2:latest"  # Default Ollama model for text cleanup
LLM_TEMPERATURE = 0.0            # Low = deterministic, high = creative
OLLAMA_URL      = "http://localhost:11434"
# Models offered in the tray picker (name -> description for tooltip)
LLM_MODEL_OPTIONS = [
    "llama3.2:latest",
    "llama3.2:1b",
    "gemma3:4b",
    "gemma3:12b",
    "phi4:latest",
    "qwen2.5-coder:14b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "granite3.2:latest",
]

# LLM prompt personalities
LLM_PROMPTS = {
    "Minimal": (
        "TASK: Fix punctuation and capitalization in this speech transcription.\n"
        "RULES: Do not rephrase, summarize, or remove any words. Do not add any "
        "commentary, explanation, or prefix. Output ONLY the corrected text and "
        "absolutely nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "Natural": (
        "TASK: Clean up this speech transcription into natural written text.\n"
        "RULES: Fix capitalization and punctuation. Remove filler words (um, uh, like, "
        "you know) and false starts. Preserve the speaker's intended meaning. Do not add "
        "any commentary, explanation, or prefix. Output ONLY the corrected text and "
        "absolutely nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "Professional": (
        "TASK: Rewrite this speech transcription as clean, professional text.\n"
        "RULES: Fix punctuation, capitalization, and grammar. Remove filler words and "
        "verbal tics. Lightly restructure run-on sentences for clarity. Keep the original "
        "meaning and tone. Do not add any commentary, explanation, or prefix. Output ONLY "
        "the corrected text and absolutely nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "Bullet Points": (
        "TASK: Convert this speech transcription into a bullet-point list.\n"
        "RULES: Extract each distinct thought or idea as its own bullet point. "
        "Each bullet should be a clear, complete sentence that makes sense on its own. "
        "Lightly rephrase for readability if needed but preserve the original meaning. "
        "Fix punctuation and capitalization. Use a dash (-) for each bullet. Do not add "
        "any commentary, explanation, or prefix. Output ONLY the bullet list and "
        "absolutely nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "Concise": (
        "TASK: Condense this speech transcription to its shortest form.\n"
        "RULES: Remove all filler, redundancy, and unnecessary words. Keep the full "
        "meaning intact. Fix punctuation and capitalization. Do not add any commentary, "
        "explanation, or prefix. Output ONLY the condensed text and absolutely nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "Story": (
        "TASK: Transform this speech into a short, whimsical micro-story.\n"
        "RULES: Take the speaker's words and weave them into a fun, imaginative narrative "
        "with a beginning, middle, and end. Be creative, playful, and dramatic. Add vivid "
        "details and flair. Keep it to 2-4 sentences maximum — punchy and tight. "
        "Do not add any commentary or prefix. Output ONLY the story and absolutely "
        "nothing else.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
}
LLM_DEFAULT_PROMPT = "Minimal"
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
        self._on_select = None  # callback(text) when user clicks a debug result
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

        # Side-by-side comparison labels for LLM cleanup
        self._compare_frame = tk.Frame(self._frame, bg=self.TRANSPARENT)
        self._orig_label = tk.Label(
            self._compare_frame,
            text="",
            font=("Segoe UI", 14),
            padx=12,
            pady=8,
            wraplength=350,
            justify="left",
        )
        self._arrow_label = tk.Label(
            self._compare_frame,
            text="\u2192",
            font=("Segoe UI", 20, "bold"),
            padx=8,
            pady=8,
            bg=self.TRANSPARENT,
            fg="#FFFFFF",
        )
        self._clean_label = tk.Label(
            self._compare_frame,
            text="",
            font=("Segoe UI", 16, "bold"),
            padx=12,
            pady=8,
            wraplength=350,
            justify="left",
        )

        # Debug mode: 3-column display for all prompt personalities
        self._debug_frame = tk.Frame(self._frame, bg=self.TRANSPARENT)
        self._debug_labels = {}
        self._debug_colors = {
            "Minimal": "#42A5F5",        # blue
            "Natural": "#66BB6A",        # green
            "Professional": "#AB47BC",   # purple
            "Bullet Points": "#FFA726",  # orange
            "Concise": "#26C6DA",        # cyan
            "Story": "#FF7043",          # coral
        }

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

    def show(self, text, style="recording", duration=None, original=None, debug_results=None):
        def _update():
            self._cancel_animations()
            self._current_style = style

            self._mic_label.pack_forget()
            self._label.pack_forget()
            self._compare_frame.pack_forget()
            self._debug_frame.pack_forget()

            if style == "debug" and debug_results is not None:
                # Show original + all 3 prompt results as clickable columns
                self._root.config(bg=self.TRANSPARENT)
                self._frame.config(bg=self.TRANSPARENT)
                self._root.attributes("-transparentcolor", self.TRANSPARENT)

                # Clear old debug labels
                for w in self._debug_frame.winfo_children():
                    w.destroy()

                # Original text header
                orig_header = tk.Label(
                    self._debug_frame, text="Original",
                    font=("Segoe UI", 10, "bold"), bg="#222222", fg="#888888",
                    padx=8, pady=2,
                )
                orig_header.grid(row=0, column=0, columnspan=len(debug_results), sticky="w", padx=4, pady=(4, 0))
                orig_text = tk.Label(
                    self._debug_frame, text=original or text,
                    font=("Segoe UI", 11), bg="#222222", fg="#AAAAAA",
                    padx=8, pady=4, wraplength=900, justify="left",
                )
                orig_text.grid(row=1, column=0, columnspan=len(debug_results), sticky="we", padx=4, pady=(0, 8))

                hint = tk.Label(
                    self._debug_frame, text="\u2190 \u2192 to navigate, Enter to select, or click",
                    font=("Segoe UI", 9), bg="#222222", fg="#666666",
                    padx=8, pady=0,
                )
                hint.grid(row=4, column=0, columnspan=len(debug_results), pady=(4, 2))

                # Build columns and track them for keyboard navigation
                columns = []  # list of (col_frame, header, body, result_text)
                for col, (name, result_text) in enumerate(debug_results.items()):
                    color = self._debug_colors.get(name, "#FFFFFF")
                    col_frame = tk.Frame(self._debug_frame, bg="#333333", cursor="hand2")
                    col_frame.grid(row=3, column=col, sticky="nswe", padx=4, pady=(0, 4))

                    header = tk.Label(
                        col_frame, text=name,
                        font=("Segoe UI", 9, "bold"), bg="#333333", fg=color,
                        padx=6, pady=2, cursor="hand2",
                    )
                    header.pack(anchor="w")
                    body = tk.Label(
                        col_frame, text=result_text,
                        font=("Segoe UI", 10), bg="#333333", fg=color,
                        padx=6, pady=4, wraplength=170, justify="left", anchor="nw",
                        cursor="hand2",
                    )
                    body.pack(anchor="w", fill="x")
                    columns.append((col_frame, header, body, result_text))

                    # Click handler
                    def _on_click(e, txt=result_text):
                        if self._on_select:
                            self._on_select(txt)
                        self._do_hide()

                    # Hover effects
                    def _on_enter(e, f=col_frame, h=header, b=body):
                        f.config(bg="#444444")
                        h.config(bg="#444444")
                        b.config(bg="#444444")

                    def _on_leave(e, f=col_frame, h=header, b=body):
                        f.config(bg="#333333")
                        h.config(bg="#333333")
                        b.config(bg="#333333")

                    for widget in (col_frame, header, body):
                        widget.bind("<Button-1>", _on_click)
                        widget.bind("<Enter>", _on_enter)
                        widget.bind("<Leave>", _on_leave)

                # Track columns for external keyboard navigation
                self._debug_columns = columns
                self._debug_selected = -1

                self._debug_frame.pack(pady=4)
                # No auto-dismiss — user must pick a result
                duration = None

            elif style == "compare" and original is not None:
                # Side-by-side: original (left) -> cleaned (right)
                self._root.config(bg=self.TRANSPARENT)
                self._frame.config(bg=self.TRANSPARENT)
                self._root.attributes("-transparentcolor", self.TRANSPARENT)

                self._orig_label.config(text=original, bg="#333333", fg="#AAAAAA")
                self._arrow_label.config(bg=self.TRANSPARENT, fg="#FFFFFF")
                self._clean_label.config(text=text, bg="#333333", fg="#4CAF50")

                self._orig_label.pack(side="left", padx=(8, 0), pady=4)
                self._arrow_label.pack(side="left", padx=4, pady=4)
                self._clean_label.pack(side="left", padx=(0, 8), pady=4)
                self._compare_frame.pack(pady=4)

            elif style == "result":
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

    def debug_navigate(self, direction):
        """Move highlight left (-1) or right (+1) in Multi-Style Preview."""
        def _update():
            cols = getattr(self, "_debug_columns", [])
            if not cols:
                return
            sel = getattr(self, "_debug_selected", -1)
            # Un-highlight previous
            if 0 <= sel < len(cols):
                pf, ph, pb, _ = cols[sel]
                pf.config(bg="#333333")
                ph.config(bg="#333333")
                pb.config(bg="#333333")
            # Compute new index
            if direction > 0:
                new = min(sel + 1, len(cols) - 1) if sel >= 0 else 0
            else:
                new = max(sel - 1, 0)
            self._debug_selected = new
            f, h, b, _ = cols[new]
            f.config(bg="#444444")
            h.config(bg="#444444")
            b.config(bg="#444444")
        self._root.after(0, _update)

    def debug_confirm(self):
        """Confirm the currently highlighted Multi-Style Preview selection."""
        def _update():
            cols = getattr(self, "_debug_columns", [])
            sel = getattr(self, "_debug_selected", -1)
            if 0 <= sel < len(cols):
                _, _, _, txt = cols[sel]
                if self._on_select:
                    self._on_select(txt)
                self._do_hide()
        self._root.after(0, _update)

    def debug_dismiss(self):
        """Dismiss Multi-Style Preview without selecting."""
        self._root.after(0, self._do_hide)

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

    # Config file lives next to the script
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_type.json")

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
        self._llm_cleanup = LLM_CLEANUP
        self._llm_model = LLM_MODEL
        self._llm_prompt = LLM_DEFAULT_PROMPT
        self._llm_debug = False
        self._auto_submit = False  # Press Enter after pasting from Multi-Style Preview
        # Streaming transcription state
        self._stream_text = ""
        self._stream_lock = threading.Lock()
        # Load saved settings (overrides defaults above)
        self._load_config()

    def _load_config(self):
        """Load persistent settings from JSON config file."""
        try:
            with open(self._config_path, "r") as f:
                cfg = json.load(f)
            self._hotkey = cfg.get("hotkey", self._hotkey)
            self._chime_enabled = cfg.get("chime_enabled", self._chime_enabled)
            self._llm_cleanup = cfg.get("llm_cleanup", self._llm_cleanup)
            self._llm_model = cfg.get("llm_model", self._llm_model)
            self._llm_prompt = cfg.get("llm_prompt", self._llm_prompt)
            self._llm_debug = cfg.get("llm_debug", self._llm_debug)
            self._auto_submit = cfg.get("auto_submit", self._auto_submit)
            if self._cuda_available:
                self._use_cuda = cfg.get("use_cuda", self._use_cuda)
            # Validate llm_prompt still exists
            if self._llm_prompt not in LLM_PROMPTS:
                self._llm_prompt = LLM_DEFAULT_PROMPT
            print(f"[CONF ] Loaded settings from {self._config_path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CONF ] Failed to load config: {e}")

    def _save_config(self):
        """Persist current settings to JSON config file."""
        cfg = {
            "hotkey": self._hotkey,
            "chime_enabled": self._chime_enabled,
            "use_cuda": self._use_cuda,
            "llm_cleanup": self._llm_cleanup,
            "llm_model": self._llm_model,
            "llm_prompt": self._llm_prompt,
            "llm_debug": self._llm_debug,
            "auto_submit": self._auto_submit,
        }
        try:
            with open(self._config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"[CONF ] Failed to save config: {e}")

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

    def _call_ollama(self, prompt_text):
        """Send a prompt to Ollama and return the response text."""
        payload = json.dumps({
            "model": self._llm_model,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": LLM_TEMPERATURE, "num_predict": 512},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result.get("response", "").strip()
        # Strip wrapping quotes the model sometimes adds
        if len(text) >= 2 and text[0] in ('"', '\u201c') and text[-1] in ('"', '\u201d'):
            text = text[1:-1].strip()
        return text

    def cleanup_text(self, text, prompt_name=None):
        """Send text to Ollama for cleanup using the selected prompt personality."""
        name = prompt_name or self._llm_prompt
        prompt_template = LLM_PROMPTS[name]
        prompt = prompt_template.format(text=text)
        try:
            cleaned = self._call_ollama(prompt)
            return cleaned if cleaned else text
        except Exception as e:
            print(f"[LLM  ] Cleanup failed ({name}): {e}")
            return text

    def cleanup_text_all(self, text):
        """Run all prompt personalities in parallel. Returns dict of {name: result}."""
        results = {}
        lock = threading.Lock()

        def _run(name):
            result = self.cleanup_text(text, prompt_name=name)
            with lock:
                results[name] = result

        threads = [threading.Thread(target=_run, args=(name,)) for name in LLM_PROMPTS]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6)
        # Fill in any that timed out
        for name in LLM_PROMPTS:
            if name not in results:
                results[name] = "(timed out)"
        return results

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
            raw_text = self.transcribe_streaming(audio_np)
            if raw_text:
                print(f"[TEXT ] {raw_text}")

                if self._llm_debug:
                    # Multi-Style Preview: run all prompts, let user pick
                    import keyboard as kb
                    all_results = self.cleanup_text_all(raw_text)
                    for name, result in all_results.items():
                        print(f"[LLM  ] {name}: {result}")
                    # Save the foreground window so we can restore it on select
                    user32 = ctypes.windll.user32
                    saved_hwnd = user32.GetForegroundWindow()
                    nav_hooks = []

                    def _cleanup_hooks():
                        for h in nav_hooks:
                            kb.unhook(h)
                        nav_hooks.clear()

                    def _on_debug_select(selected_text):
                        import pyautogui
                        _cleanup_hooks()
                        print(f"[LLM  ] Selected: {selected_text}")
                        user32.SetForegroundWindow(saved_hwnd)
                        time.sleep(0.1)
                        self.paste_text(selected_text)
                        if self._auto_submit:
                            time.sleep(0.05)
                            pyautogui.press("enter")

                    def _on_nav_key(e):
                        if e.event_type != "down":
                            return
                        if e.name in ("right", "down"):
                            self.bubble.debug_navigate(1)
                        elif e.name in ("left", "up"):
                            self.bubble.debug_navigate(-1)
                        elif e.name == "enter":
                            self.bubble.debug_confirm()
                        elif e.name == "esc":
                            _cleanup_hooks()
                            self.bubble.debug_dismiss()

                    nav_hooks.append(kb.hook(_on_nav_key, suppress=True))
                    self.bubble._on_select = _on_debug_select
                    self.play_chime(CHIME_DONE)
                    self.bubble.show(None, style="debug", original=raw_text, debug_results=all_results)
                elif self._llm_cleanup:
                    cleaned = self.cleanup_text(raw_text)
                    if cleaned != raw_text:
                        print(f"[LLM  ] {cleaned}")
                    final_text = cleaned
                    self.play_chime(CHIME_DONE)
                    self.paste_text(final_text)
                    if final_text != raw_text and BUBBLE_DURATION > 0:
                        self.bubble.show(final_text, style="compare", duration=BUBBLE_DURATION, original=raw_text)
                    elif BUBBLE_DURATION > 0:
                        self.bubble.show(final_text, style="result", duration=BUBBLE_DURATION)
                    else:
                        self.bubble.hide()
                else:
                    final_text = raw_text
                    self.play_chime(CHIME_DONE)
                    self.paste_text(final_text)
                    if BUBBLE_DURATION > 0:
                        self.bubble.show(final_text, style="result", duration=BUBBLE_DURATION)
                    else:
                        self.bubble.hide()
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
        self._save_config()
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
        self._save_config()
        threading.Thread(target=self.reload_model, daemon=True).start()

    def _toggle_chime(self):
        self._chime_enabled = not self._chime_enabled
        state = "on" if self._chime_enabled else "off"
        print(f"[CHIME] Sound chimes {state}")
        self._save_config()

    def _get_installed_models(self):
        """Query Ollama for locally available models."""
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def _pull_model(self, model_name):
        """Pull a model from Ollama, showing progress in the bubble."""
        import subprocess
        self.bubble.show(f"Pulling {model_name}...", style="transcribing")
        print(f"[LLM  ] Pulling model: {model_name}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                print(f"[LLM  ] Pull complete: {model_name}")
                self.bubble.show(f"{model_name} ready!", style="result", duration=2)
                return True
            else:
                print(f"[LLM  ] Pull failed: {result.stderr.strip()}")
                self.bubble.show(f"Pull failed: {model_name}", style="transcribing", duration=3)
                return False
        except subprocess.TimeoutExpired:
            print(f"[LLM  ] Pull timed out: {model_name}")
            self.bubble.show(f"Pull timed out: {model_name}", style="transcribing", duration=3)
            return False
        except Exception as e:
            print(f"[LLM  ] Pull error: {e}")
            self.bubble.show(f"Pull error: {model_name}", style="transcribing", duration=3)
            return False

    def _set_llm_model(self, model_name):
        """Switch to a different Ollama model, pulling if needed."""
        def _do_switch():
            installed = self._get_installed_models()
            if model_name not in installed:
                success = self._pull_model(model_name)
                if not success:
                    return
            self._llm_model = model_name
            print(f"[LLM  ] Model: {model_name}")
            self._save_config()
            self._rebuild_tray_menu()

        threading.Thread(target=_do_switch, daemon=True).start()

    def _toggle_llm_cleanup(self):
        self._llm_cleanup = not self._llm_cleanup
        state = "on" if self._llm_cleanup else "off"
        print(f"[LLM  ] Text cleanup {state}")
        self._save_config()
        self._rebuild_tray_menu()

    def _set_llm_prompt(self, name):
        self._llm_prompt = name
        print(f"[LLM  ] Prompt style: {name}")
        self._save_config()
        self._rebuild_tray_menu()

    def _toggle_llm_debug(self):
        self._llm_debug = not self._llm_debug
        state = "on" if self._llm_debug else "off"
        print(f"[LLM  ] Multi-Style Preview {state}")
        self._save_config()
        self._rebuild_tray_menu()

    def _toggle_auto_submit(self):
        self._auto_submit = not self._auto_submit
        state = "on" if self._auto_submit else "off"
        print(f"[LLM  ] Auto-submit {state}")
        self._save_config()
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

        # LLM submenu: toggle, prompt style picker, debug mode
        def _make_prompt_handler(name):
            return lambda icon, item: self._set_llm_prompt(name)

        prompt_items = [
            pystray.MenuItem(
                name,
                _make_prompt_handler(name),
                checked=lambda item, n=name: n == self._llm_prompt,
                radio=True,
            )
            for name in LLM_PROMPTS
        ]

        # Model picker submenu
        installed = self._get_installed_models()

        def _make_model_handler(m):
            return lambda icon, item: self._set_llm_model(m)

        model_items = []
        for m in LLM_MODEL_OPTIONS:
            is_installed = m in installed
            label = m if is_installed else f"{m}  [pull]"
            model_items.append(pystray.MenuItem(
                label,
                _make_model_handler(m),
                checked=lambda item, mn=m: mn == self._llm_model,
                radio=True,
            ))

        llm_submenu = pystray.Menu(
            pystray.MenuItem(
                "Enabled",
                lambda icon, item: self._toggle_llm_cleanup(),
                checked=lambda item: self._llm_cleanup,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model", pystray.Menu(*model_items)),
            pystray.MenuItem("Style", pystray.Menu(*prompt_items)),
            pystray.MenuItem(
                "Multi-Style Preview",
                lambda icon, item: self._toggle_llm_debug(),
                checked=lambda item: self._llm_debug,
            ),
            pystray.MenuItem(
                "Auto-submit on select",
                lambda icon, item: self._toggle_auto_submit(),
                checked=lambda item: self._auto_submit,
            ),
        )

        menu = pystray.Menu(
            pystray.MenuItem(f"Hold {self._hotkey.upper()} to record", lambda: None, enabled=False),
            pystray.MenuItem(f"Model: {WHISPER_MODEL}", lambda: None, enabled=False),
            device_item,
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Hotkey", pystray.Menu(*hotkey_items)),
            chime_item,
            pystray.MenuItem(f"LLM ({self._llm_model})", llm_submenu),
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

        # Start audio stream
        default_dev = sd.query_devices(kind='input')
        print(f"[MIC  ] Using: {default_dev['name']} (channels={default_dev['max_input_channels']})")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
            blocksize=1024,
        )
        self.stream.start()

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
