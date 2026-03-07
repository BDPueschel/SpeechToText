"""
Test suite for whisper_type.py - validates bug fixes for hanging and performance.

Mocks external dependencies (audio, whisper, Ollama, GUI) to test the logic
without requiring hardware or running services.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import numpy as np


# ---------------------------------------------------------------------------
#  Mock heavier imports before importing whisper_type
# ---------------------------------------------------------------------------
import sys

# Stub out modules that need hardware / GUI
for mod_name in ["sounddevice", "keyboard", "pyautogui", "pyperclip",
                 "pystray", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageTk"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Stub tkinter so Bubble doesn't actually open a window
_real_tk = None
try:
    import tkinter as _real_tk_mod
    _real_tk = _real_tk_mod
except ImportError:
    pass

# We need tkinter importable but won't actually run mainloop
# The Bubble class starts a thread + mainloop; we'll mock it entirely.


class MockBubble:
    """Lightweight stand-in for the Bubble overlay."""
    def __init__(self, fft_callback=None):
        self.shown = []
        self._on_select = None
        self._root = MagicMock()
        self._debug_columns = []
        self._debug_selected = -1
        self._debug_colors = {}

    def show(self, text, style="recording", duration=None, original=None, debug_results=None):
        self.shown.append({"text": text, "style": style, "duration": duration,
                           "original": original, "debug_results": debug_results})

    def hide(self):
        self.shown.append({"text": None, "style": "hide"})

    def destroy(self):
        pass

    def debug_navigate(self, d):
        pass

    def debug_confirm(self, shift_held=False):
        pass

    def debug_dismiss(self):
        pass


# ---------------------------------------------------------------------------
#  Import the module under test (after mocking deps)
# ---------------------------------------------------------------------------
import whisper_type
from whisper_type import WhisperTray


def _make_app(**config_overrides):
    """Create a WhisperTray with mocked externals and optional config."""
    with patch.object(WhisperTray, '_load_config'), \
         patch.object(WhisperTray, '_load_history'):
        app = WhisperTray()

    app.bubble = MockBubble()
    app.tray = MagicMock()
    app._sd = MagicMock()
    app.model = MagicMock()
    app._history = []
    app._history_path = "test_history.json"

    # Apply config overrides
    for k, v in config_overrides.items():
        setattr(app, k, v)

    return app


# ===================================================================
#  1. Transcription parameter tests
# ===================================================================
class TestTranscribeParams(unittest.TestCase):
    """Verify transcribe calls use beam_size=1, vad_filter, language, etc."""

    def _capture_transcribe_call(self, whisper_model_name):
        app = _make_app(_whisper_model=whisper_model_name)

        captured = {}
        def fake_transcribe(path, **kwargs):
            captured.update(kwargs)
            return iter([]), None  # (segments, info)

        app.model.transcribe = fake_transcribe
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        app.transcribe(audio)
        return captured

    def test_beam_size_is_1(self):
        params = self._capture_transcribe_call("base.en")
        self.assertEqual(params["beam_size"], 1,
                         "beam_size should be 1 for fast push-to-talk")

    def test_vad_filter_enabled(self):
        params = self._capture_transcribe_call("base.en")
        self.assertTrue(params["vad_filter"],
                        "vad_filter should be True to skip silence")

    def test_condition_on_previous_text_disabled(self):
        params = self._capture_transcribe_call("base.en")
        self.assertFalse(params["condition_on_previous_text"],
                         "condition_on_previous_text should be False to prevent hallucination loops")

    def test_language_en_for_en_models(self):
        for model in ["tiny.en", "base.en", "small.en", "medium.en"]:
            params = self._capture_transcribe_call(model)
            self.assertEqual(params["language"], "en",
                             f"language should be 'en' for {model}")

    def test_language_none_for_multilingual_models(self):
        params = self._capture_transcribe_call("large-v3")
        self.assertIsNone(params["language"],
                          "language should be None for multilingual models")


class TestTranscribeStreamingParams(unittest.TestCase):
    """Same checks for the streaming transcription path."""

    def _capture_streaming_call(self, whisper_model_name):
        app = _make_app(_whisper_model=whisper_model_name, _bubble_duration=0)

        captured = {}
        def fake_transcribe(path, **kwargs):
            captured.update(kwargs)
            return iter([]), None

        app.model.transcribe = fake_transcribe
        audio = np.zeros(16000, dtype=np.float32)
        app.transcribe_streaming(audio)
        return captured

    def test_beam_size_is_1(self):
        params = self._capture_streaming_call("base.en")
        self.assertEqual(params["beam_size"], 1)

    def test_vad_filter_enabled(self):
        params = self._capture_streaming_call("base.en")
        self.assertTrue(params["vad_filter"])

    def test_condition_on_previous_text_disabled(self):
        params = self._capture_streaming_call("base.en")
        self.assertFalse(params["condition_on_previous_text"])

    def test_language_en_for_en_models(self):
        params = self._capture_streaming_call("base.en")
        self.assertEqual(params["language"], "en")

    def test_language_none_for_multilingual(self):
        params = self._capture_streaming_call("large-v3")
        self.assertIsNone(params["language"])


# ===================================================================
#  2. Audio stream close uses abort() not stop()
# ===================================================================
class TestAudioStreamClose(unittest.TestCase):
    """Verify _close_audio_stream uses abort() to avoid blocking."""

    def test_close_calls_abort(self):
        app = _make_app()
        mock_stream = MagicMock()
        app.stream = mock_stream

        app._close_audio_stream()

        mock_stream.abort.assert_called_once()
        mock_stream.stop.assert_not_called()
        mock_stream.close.assert_called_once()
        self.assertIsNone(app.stream)

    def test_close_noop_when_no_stream(self):
        app = _make_app()
        app.stream = None
        app._close_audio_stream()  # Should not raise

    def test_close_handles_exception(self):
        app = _make_app()
        mock_stream = MagicMock()
        mock_stream.abort.side_effect = Exception("device error")
        app.stream = mock_stream

        app._close_audio_stream()  # Should not raise
        self.assertIsNone(app.stream)


# ===================================================================
#  3. Privacy mic toggle doesn't block
# ===================================================================
class TestPrivacyMicToggle(unittest.TestCase):
    """Privacy mic on/off should run stream ops on background threads."""

    def test_open_stream_on_key_down_when_privacy_on(self):
        app = _make_app(_privacy_mic=True)
        app.stream = None
        mock_stream = MagicMock()
        app._sd.InputStream.return_value = mock_stream

        # Simulate key down
        with patch.dict("sys.modules", {"keyboard": MagicMock()}) as mock_kb:
            app.on_key_down()

        self.assertTrue(app.is_recording)
        app._sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_close_stream_on_key_up_when_privacy_on(self):
        app = _make_app(_privacy_mic=True)
        mock_stream = MagicMock()
        app.stream = mock_stream
        app.is_recording = True
        app.start_time = time.time() - 0.1  # Short recording, will skip transcription
        app.audio_buffer = []

        with patch.dict("sys.modules", {"keyboard": MagicMock()}) as mock_kb:
            app.on_key_up()

        mock_stream.abort.assert_called_once()
        self.assertIsNone(app.stream)


# ===================================================================
#  4. Ollama unload uses stream: false
# ===================================================================
class TestOllamaUnload(unittest.TestCase):
    """_unload_ollama_model must send stream:false to avoid hanging."""

    @patch.object(WhisperTray, '_ollama_post')
    def test_unload_sends_stream_false(self, mock_post):
        mock_post.return_value = {}
        app = _make_app(_llm_model="llama3.2:latest")

        app._unload_ollama_model()

        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertFalse(payload["stream"],
                         "unload must send stream:false to prevent hanging")
        self.assertEqual(payload["keep_alive"], 0)
        self.assertEqual(payload["model"], "llama3.2:latest")

    @patch.object(WhisperTray, '_ollama_post')
    def test_unload_reads_response(self, mock_post):
        """_ollama_post reads and closes connection internally."""
        mock_post.return_value = {}
        app = _make_app(_llm_model="llama3.2:latest")

        app._unload_ollama_model()

        mock_post.assert_called_once()

    @patch.object(WhisperTray, '_ollama_post')
    def test_unload_handles_failure(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        app = _make_app(_llm_model="llama3.2:latest")
        app._unload_ollama_model()  # Should not raise


# ===================================================================
#  5. Ollama call_ollama timeout is reasonable
# ===================================================================
class TestOllamaTimeout(unittest.TestCase):

    @patch.object(WhisperTray, '_ollama_post')
    def test_call_ollama_uses_ollama_post(self, mock_post):
        """_call_ollama must use _ollama_post (http.client, not urllib)."""
        mock_post.return_value = {"response": "hello"}
        app = _make_app(_llm_model="llama3.2:latest")

        result = app._call_ollama("test prompt")

        mock_post.assert_called_once()
        self.assertEqual(result, "hello")

    @patch.object(WhisperTray, '_ollama_post')
    def test_call_ollama_sends_stream_false(self, mock_post):
        mock_post.return_value = {"response": "hello"}
        app = _make_app(_llm_model="llama3.2:latest")

        app._call_ollama("test prompt")

        payload = mock_post.call_args[0][1]
        self.assertFalse(payload["stream"])


# ===================================================================
#  6. Warm-up model on LLM enable
# ===================================================================
class TestWarmOllamaModel(unittest.TestCase):

    @patch.object(WhisperTray, '_call_ollama')
    def test_warm_shows_loading_then_ready(self, mock_call):
        mock_call.return_value = "hello"
        app = _make_app(_llm_model="llama3.2:latest")

        app._warm_ollama_model()

        # Should show loading, then ready
        styles = [s["style"] for s in app.bubble.shown]
        self.assertIn("transcribing", styles, "Should show loading toast")
        self.assertIn("result", styles, "Should show ready toast")

    @patch.object(WhisperTray, '_call_ollama')
    def test_warm_shows_error_on_failure(self, mock_call):
        mock_call.side_effect = Exception("connection refused")
        app = _make_app(_llm_model="llama3.2:latest")

        app._warm_ollama_model()

        # Should show loading, then error
        last = app.bubble.shown[-1]
        self.assertEqual(last["style"], "transcribing")
        self.assertIn("unavailable", last["text"].lower())

    @patch.object(WhisperTray, '_call_ollama')
    def test_warm_does_not_block_caller(self, mock_call):
        """Warm-up should complete in reasonable time even if Ollama is slow."""
        def slow_ollama(prompt):
            time.sleep(0.5)
            return "ok"
        mock_call.side_effect = slow_ollama
        app = _make_app(_llm_model="llama3.2:latest")

        t = threading.Thread(target=app._warm_ollama_model)
        t.start()
        t.join(timeout=3)
        self.assertFalse(t.is_alive(), "Warm-up should complete within timeout")


# ===================================================================
#  7. LLM toggle triggers warm/unload on background thread
# ===================================================================
class TestLLMToggle(unittest.TestCase):

    @patch.object(WhisperTray, '_warm_ollama_model')
    @patch.object(WhisperTray, '_rebuild_tray_menu')
    @patch.object(WhisperTray, '_save_config')
    def test_toggle_on_warms_model(self, mock_save, mock_rebuild, mock_warm):
        app = _make_app(_llm_cleanup=False)

        app._llm_cleanup = False
        app._toggle_llm_cleanup()

        self.assertTrue(app._llm_cleanup)
        # Give the thread a moment to start
        time.sleep(0.1)
        mock_warm.assert_called_once()

    @patch.object(WhisperTray, '_unload_ollama_model')
    @patch.object(WhisperTray, '_rebuild_tray_menu')
    @patch.object(WhisperTray, '_save_config')
    def test_toggle_off_unloads_model(self, mock_save, mock_rebuild, mock_unload):
        app = _make_app(_llm_cleanup=True)

        app._llm_cleanup = True
        app._toggle_llm_cleanup()

        self.assertFalse(app._llm_cleanup)
        time.sleep(0.1)
        mock_unload.assert_called_once()

    @patch.object(WhisperTray, '_unload_ollama_model')
    @patch.object(WhisperTray, '_rebuild_tray_menu')
    @patch.object(WhisperTray, '_save_config')
    def test_toggle_off_does_not_block(self, mock_save, mock_rebuild, mock_unload):
        """Toggling LLM off must return immediately (not block on Ollama)."""
        def slow_unload():
            time.sleep(2)
        mock_unload.side_effect = slow_unload
        app = _make_app(_llm_cleanup=True)
        app._llm_cleanup = True

        start = time.time()
        app._toggle_llm_cleanup()
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.5,
                        "Toggle should return immediately; unload runs in background")


# ===================================================================
#  8. Escape key hook cleanup
# ===================================================================
class TestEscapeKeyHook(unittest.TestCase):

    def test_on_key_down_unhooks_esc_before_registering(self):
        """Escape hook should be cleaned up before re-registering to avoid leaks."""
        app = _make_app(_privacy_mic=False)
        app.stream = MagicMock()

        mock_kb = MagicMock()
        call_log = []
        mock_kb.unhook_key = lambda k: call_log.append(("unhook", k))
        mock_kb.on_press_key = lambda k, cb, **kw: call_log.append(("register", k))

        with patch.dict("sys.modules", {"keyboard": mock_kb}):
            app.is_recording = False
            app.on_key_down()
            app.is_recording = False
            app.on_key_down()

        unhook_calls = [c for c in call_log if c[0] == "unhook" and c[1] == "esc"]
        register_calls = [c for c in call_log if c[0] == "register" and c[1] == "esc"]
        self.assertGreaterEqual(len(unhook_calls), 2,
                                f"Should unhook esc before each re-registration, got: {call_log}")
        self.assertGreaterEqual(len(register_calls), 2,
                                f"Should register esc handler each time, got: {call_log}")


# ===================================================================
#  9. Full recording -> transcribe flow (no LLM)
# ===================================================================
class TestRecordingFlow(unittest.TestCase):

    def test_short_press_skips_transcription(self):
        app = _make_app(_privacy_mic=False, _bubble_duration=0)
        app.stream = MagicMock()

        with patch.dict("sys.modules", {"keyboard": MagicMock()}):
            app.on_key_down()
            time.sleep(0.05)
            app.start_time = time.time()  # Reset to make it < 0.3s
            app.audio_buffer = []
            app.on_key_up()

        app.model.transcribe.assert_not_called()

    def test_normal_recording_triggers_transcription(self):
        app = _make_app(_privacy_mic=False, _llm_cleanup=False,
                        _bubble_duration=0, _tone_detect=False)
        app.stream = MagicMock()

        # Fake a segment
        mock_seg = MagicMock()
        mock_seg.text = "hello world"
        app.model.transcribe.return_value = (iter([mock_seg]), None)

        with patch.dict("sys.modules", {"keyboard": MagicMock()}), \
             patch.object(app, 'paste_text') as mock_paste, \
             patch.object(app, '_add_history'):
            app.on_key_down()
            # Simulate 0.5s of audio
            app.start_time = time.time() - 0.5
            app.audio_buffer = [np.zeros(8000, dtype=np.float32)]
            app.on_key_up()

            # Wait for background transcription thread
            time.sleep(1)

        mock_paste.assert_called_once_with("hello world")


# ===================================================================
#  10. Full recording -> transcribe flow (with LLM cleanup)
# ===================================================================
class TestRecordingFlowWithLLM(unittest.TestCase):

    @patch.object(WhisperTray, '_call_ollama')
    def test_llm_cleanup_applied(self, mock_ollama):
        mock_ollama.return_value = "Hello, world!"
        app = _make_app(_privacy_mic=False, _llm_cleanup=True,
                        _llm_debug=False, _bubble_duration=0,
                        _tone_detect=False, _translate_lang=None)
        app.stream = MagicMock()

        mock_seg = MagicMock()
        mock_seg.text = "hello world"
        app.model.transcribe.return_value = (iter([mock_seg]), None)

        with patch.dict("sys.modules", {"keyboard": MagicMock()}), \
             patch.object(app, 'paste_text') as mock_paste, \
             patch.object(app, '_add_history'):
            app.on_key_down()
            app.start_time = time.time() - 0.5
            app.audio_buffer = [np.zeros(8000, dtype=np.float32)]
            app.on_key_up()
            time.sleep(1)

        mock_paste.assert_called_once_with("Hello, world!")


# ===================================================================
#  11. Bubble duration - hide timer is set for non-debug styles
# ===================================================================
class TestBubbleDuration(unittest.TestCase):
    """The show() closure must not shadow `duration` with a local assignment."""

    def test_duration_variable_not_shadowed(self):
        """Regression: `duration = None` in debug branch must not make
        `duration` a local variable in _update, breaking all other styles."""
        import ast, inspect, textwrap
        from whisper_type import Bubble

        source = inspect.getsource(Bubble.show)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Find the inner _update function
        show_func = tree.body[0]
        update_func = None
        for node in ast.walk(show_func):
            if isinstance(node, ast.FunctionDef) and node.name == "_update":
                update_func = node
                break

        self.assertIsNotNone(update_func, "Could not find _update in show()")

        # Check that _update does NOT assign to `duration`
        for node in ast.walk(update_func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "duration":
                        self.fail(
                            "Found `duration = ...` inside _update closure. "
                            "This shadows the outer `duration` parameter and breaks "
                            "the hide timer for all non-debug styles."
                        )


if __name__ == "__main__":
    unittest.main()
