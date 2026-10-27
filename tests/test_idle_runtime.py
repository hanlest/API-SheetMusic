"""Tests for idle VRAM unload runtime (no real weights)."""

import time
from unittest.mock import MagicMock, patch

from muscriptor.idle_runtime import IdleModelRuntime


def _make_runtime(idle_seconds: float = 0.05) -> IdleModelRuntime:
    return IdleModelRuntime(
        weights_path=None,
        device="cpu",
        dtype=None,
        idle_seconds=idle_seconds,
        poll_seconds=0.01,
    )


def test_unload_after_idle_without_inflight():
    runtime = _make_runtime()
    mock_model = MagicMock()
    mock_model.weights_loaded = True
    runtime._model = mock_model
    runtime._last_used = time.monotonic() - 1.0
    runtime._inflight = 0

    assert runtime.unload_if_idle() is True
    mock_model.release_accelerator_memory.assert_called_once()


def test_no_unload_while_transcription_inflight():
    runtime = _make_runtime()
    mock_model = MagicMock()
    mock_model.weights_loaded = True
    runtime._model = mock_model
    runtime._last_used = time.monotonic() - 1.0
    runtime._inflight = 1

    assert runtime.unload_if_idle() is False
    mock_model.release_accelerator_memory.assert_not_called()


def test_lazy_load_on_transcribe_and_postprocess():
    runtime = _make_runtime(idle_seconds=60.0)
    fake = MagicMock()
    fake.transcribe_and_postprocess.return_value = (b"mid", None)

    with patch(
        "muscriptor.idle_runtime.TranscriptionModel.load_model", return_value=fake
    ) as load:
        out = runtime.transcribe_and_postprocess(("wav", 16000))

    load.assert_called_once()
    assert out == (b"mid", None)
    fake.transcribe_and_postprocess.assert_called_once()
