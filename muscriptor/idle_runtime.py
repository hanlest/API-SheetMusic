"""Carga diferida y liberación de VRAM tras inactividad en el servidor HTTP."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.beats import BeatGrid, TempoDetection
from muscriptor.utils.midi import ExportOptions

log = logging.getLogger(__name__)

_DEFAULT_IDLE_SECONDS = 60.0
_DEFAULT_POLL_SECONDS = 5.0


def idle_unload_seconds_from_env() -> float:
    raw = os.environ.get("MUSCRIPTOR_IDLE_UNLOAD_SECONDS", str(_DEFAULT_IDLE_SECONDS))
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "MUSCRIPTOR_IDLE_UNLOAD_SECONDS=%r no es un número; uso %.0fs.",
            raw,
            _DEFAULT_IDLE_SECONDS,
        )
        return _DEFAULT_IDLE_SECONDS


class IdleModelRuntime:
    """Envuelve ``TranscriptionModel`` con carga bajo demanda y unload por inactividad."""

    def __init__(
        self,
        *,
        weights_path: str | Path | None,
        device: str | None,
        dtype: str | None,
        idle_seconds: float = _DEFAULT_IDLE_SECONDS,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        self._weights_path = weights_path
        self._device = device
        self._dtype = dtype
        self._idle_seconds = idle_seconds
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._model: TranscriptionModel | None = None
        self._export_shell: TranscriptionModel | None = None
        self._last_used = time.monotonic()
        self._inflight = 0
        self._watcher_task: asyncio.Task[None] | None = None

    @property
    def weights_loaded(self) -> bool:
        with self._lock:
            return self._model is not None and self._model.weights_loaded

    def _touch(self) -> None:
        self._last_used = time.monotonic()

    def _get_inference_model(self) -> TranscriptionModel:
        with self._lock:
            if self._model is None:
                log.info("Cargando modelo de transcripción (%s)…", self._device or "auto")
                self._model = TranscriptionModel.load_model(
                    weights_path=self._weights_path,
                    device=self._device,
                    dtype=self._dtype,
                )
            else:
                self._model.ensure_weights_loaded()
            self._touch()
            return self._model

    def _export_model(self) -> TranscriptionModel:
        with self._lock:
            if self._export_shell is None:
                self._export_shell = TranscriptionModel.create_export_shell()
            return self._export_shell

    @contextmanager
    def _work_session(self) -> Iterator[TranscriptionModel]:
        model = self._get_inference_model()
        with self._lock:
            self._inflight += 1
        try:
            yield model
        finally:
            with self._lock:
                self._inflight -= 1
                self._touch()

    def unload_if_idle(self) -> bool:
        """Libera pesos si lleva ``idle_seconds`` sin transcripciones."""
        with self._lock:
            if self._model is None or not self._model.weights_loaded:
                return False
            if self._inflight > 0:
                return False
            idle_for = time.monotonic() - self._last_used
            if idle_for < self._idle_seconds:
                return False
            log.info(
                "Sin transcripciones durante %.0fs; liberando VRAM del modelo.",
                idle_for,
            )
            self._model.release_accelerator_memory()
            return True

    async def _idle_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_seconds)
                self.unload_if_idle()
        except asyncio.CancelledError:
            return

    def lifespan(self):
        @asynccontextmanager
        async def _cm(_app: Any):
            self._watcher_task = asyncio.create_task(self._idle_loop())
            try:
                yield
            finally:
                if self._watcher_task is not None:
                    self._watcher_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._watcher_task
                    self._watcher_task = None

        return _cm

    def transcribe(self, *args: Any, **kwargs: Any) -> Iterator[
        NoteStartEvent | NoteEndEvent | ProgressEvent
    ]:
        with self._work_session() as model:
            yield from model.transcribe(*args, **kwargs)

    def transcribe_and_postprocess(self, *args: Any, **kwargs: Any) -> tuple[bytes, BeatGrid | None]:
        with self._work_session() as model:
            return model.transcribe_and_postprocess(*args, **kwargs)

    def detect_beat_grid_for(
        self, audio: Any, mode: TempoDetection = "best-effort"
    ) -> BeatGrid | None:
        with self._work_session() as model:
            return model.detect_beat_grid_for(audio, mode)

    def events_to_midi_bytes(
        self,
        events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent],
        beat_grid: BeatGrid | None = None,
        export: ExportOptions | None = None,
    ) -> bytes:
        return self._export_model().events_to_midi_bytes(events, beat_grid, export)
