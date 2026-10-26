"""FastAPI server exposing transcription as an SSE event stream.

POST /transcribe with an audio file (multipart/form-data field `file`; WAV,
or any format soundfile/libsndfile can read — mp3, flac, ogg, m4a, …) returns
`text/event-stream`. Each event's data is a JSON dict tagged by `type`:
`start` / `end` note events (same shape as `muscriptor.main._event_to_dict`),
`progress` chunk anchors (`{completed, total}`), and a final
`transcription_complete` event carrying the base64-encoded .mid file (`data`)
plus the detected `beat_grid`
(`{bpm, beats_per_bar, first_downbeat, onset_delay, beat_subdivision, steady,
beats}`, or null if no beats were found). `steady` says whether the tracked
`beats` fit the constant `bpm`; when not, the MIDI follows them with a tempo
map. `onset_delay` is how late the streamed note times are against those
beats: the MIDI has it taken out already, an SSE consumer has to subtract it.

POST /export takes JSON instead: the streamed notes, that `beat_grid`, and
`options` (`muscriptor.utils.midi.ExportOptions`: tempo/meter/downbeat
corrections, quantization, gap filling), and returns the .mid written with
them. It runs no model, so the UI can re-export as the user changes options.

POST /transcribe/midi takes the same upload but blocks until transcription
completes and returns the raw `audio/midi` bytes directly (no SSE, no
base64), with a `Content-Disposition: attachment` header. Audio longer than
15 minutes is rejected with 413.

POST /sheets takes a MIDI upload instead of audio (a quantized one from
/export, with `quantized=true`) and returns every file
`muscriptor.utils.sheets.write_sheets` engraves from it — MusicXML, the full
score, one PDF per instrument — as a single uncompressed zip. It needs
MuseScore 4+ on the server, and answers 503 when there is none.
"""

import asyncio
import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
import time
import wave
import zipfile
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Callable, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.soundfonts import SF3_URL
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file, _read_wav_file
from muscriptor.utils.beats import BeatDetectionError, BeatGrid, TempoDetection
from muscriptor.utils.download import download_if_necessary
from muscriptor.utils.midi import ExportOptions
from muscriptor.utils.sheets import (
    MuseScoreError,
    MuseScoreNotFoundError,
    write_sheets,
)


def _make_release_once(lock: threading.Lock):
    """Return a callable that releases `lock` at most once.

    Safe to call from multiple cleanup paths (generator finally + response
    background task), possibly from different threads, without risking a
    double-release RuntimeError.
    """
    guard = threading.Lock()
    released = False

    def release():
        nonlocal released
        with guard:
            if released:
                return
            released = True
        lock.release()

    return release


_MAX_TRANSCRIBE_MIDI_DURATION_S = 15 * 60

SHEETS_ZIP_NAME = "sheets.zip"

# Enum para OpenAPI/Swagger: desplegable con los nombres exactos del modelo.
InstrumentGroup = Enum(
    "InstrumentGroup",
    {name: name for name in sorted(MT3_FULL_PLUS_GROUP_NAMES)},
    type=str,
)


def _instruments_form_description() -> str:
    names_md = "\n".join(
        f"- `{name}`" for name in sorted(MT3_FULL_PLUS_GROUP_NAMES)
    )
    return (
        "Instrumentos que pueden sonar en la grabación. En multipart, repita el "
        "campo `instruments` una vez por valor (p. ej. `instruments=violin` y "
        "`instruments=drums`).\n\n"
        "**Detección automática:** no envíe el campo `instruments` (en Swagger, "
        "déjelo sin añadir filas; no use valores vacíos).\n\n"
        f"**Listado completo ({len(MT3_FULL_PLUS_GROUP_NAMES)} nombres):**\n\n"
        f"{names_md}"
    )


def _instruments_for_model(groups: list[InstrumentGroup]) -> list[str] | None:
    if not groups:
        return None
    return [g.value for g in groups]


_OPENAPI_TAGS = [
    {
        "name": "sistema",
        "description": "Comprobaciones de estado y recursos auxiliares del servidor.",
    },
    {
        "name": "transcripción",
        "description": "Convierte audio en eventos de notas, MIDI y metadatos de tempo.",
    },
    {
        "name": "exportación",
        "description": "Genera archivos MIDI o partituras a partir de una transcripción.",
    },
    {
        "name": "audio",
        "description": "Síntesis y mezcla del MIDI transcrito con el audio original.",
    },
]

_API_DESCRIPTION = """
API HTTP de **MuScriptor** para transcribir grabaciones a MIDI y partitura.

### Flujo habitual
1. **POST /transcribe** — sube audio y recibe eventos en tiempo real (SSE) más un MIDI final en base64.
2. **POST /export** — vuelve a generar el `.mid` con opciones de cuantización y tempo (sin volver a ejecutar el modelo).
3. **POST /sheets** — graba partituras (PDF, MusicXML, etc.) en un ZIP; requiere MuseScore 4+ en el servidor.

### Alternativa sin SSE
**POST /transcribe/midi** devuelve el archivo MIDI directamente (máximo 15 minutos de audio).

Documentación interactiva: **/docs** (Swagger UI). Esquema OpenAPI: **/openapi.json**.
"""


def engrave_to_zip(midi_bytes: bytes, quantized: bool = False) -> bytes:
    """Engrave `midi_bytes` and pack everything written into one zip.

    Runs `write_sheets` into a scratch directory that is thrown away once the
    archive is built, so the server keeps nothing on disk between requests.
    Member names are the bare filenames — the directory layout documented under
    "Sheet music" in the README, flattened by one level.

    Stored, not deflated: the client unpacks this archive in the browser to
    offer the files one at a time, and all but the MusicXML are PDFs, which
    carry compression of their own.
    """
    with tempfile.TemporaryDirectory(prefix="muscriptor-sheets-") as tmp:
        written = write_sheets(midi_bytes, Path(tmp), quantized=quantized)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in written:
                archive.write(path, arcname=path.name)
        return buf.getvalue()


class ExportNote(BaseModel):
    """Una nota tal como la envió /transcribe (tiempos sin restar `onset_delay`)."""

    instrument: str = Field(description="Nombre del grupo de instrumento (véase GET /instruments).")
    pitch: int = Field(ge=0, le=127, description="Altura MIDI (0–127).")
    start: float = Field(description="Inicio en segundos.")
    end: float = Field(description="Fin en segundos.")


class ExportGrid(BaseModel):
    """La rejilla de pulsos (`beat_grid`) devuelta por /transcribe, reenviada tal cual."""

    bpm: float = Field(gt=0, description="Tempo en pulsos por minuto.")
    beats_per_bar: int | None = Field(
        default=None, description="Compás detectado (pulsos por compás)."
    )
    first_downbeat: float = Field(
        default=0.0, description="Segundo del primer tiempo fuerte."
    )
    onset_delay: float = Field(
        default=0.0,
        description="Retardo en segundos entre los tiempos y las notas transmitidas.",
    )
    beat_subdivision: int | None = Field(
        default=None, description="Subdivisión por defecto al cuantizar en /export."
    )
    steady: bool = Field(
        default=True,
        description="Si los tiempos encajan con un BPM constante.",
    )
    beats: list[float] | None = Field(
        default=None, description="Posiciones en segundos de cada pulso (null si no hay)."
    )


class ExportSettings(BaseModel):
    """Opciones de exportación MIDI (`ExportOptions`)."""

    bpm: float | None = Field(default=None, ge=20, le=400, description="Tempo manual.")
    beats_per_bar: int | None = Field(
        default=None, ge=1, le=32, description="Compás manual."
    )
    first_downbeat: float | None = Field(
        default=None, description="Primer tiempo fuerte manual."
    )
    tempo_map: bool | None = Field(
        default=None, description="Usar mapa de tempo si el pulso no es constante."
    )
    beat_scale: Literal[0.25, 0.5, 1, 2, 4] = Field(
        default=1, description="Escala de la rejilla de pulsos."
    )
    quantize: bool = Field(default=False, description="Ajustar notas a la rejilla.")
    subdivision: int | None = Field(
        default=None, ge=1, le=32, description="Subdivisión al cuantizar."
    )
    strength: float = Field(
        default=1.0, ge=0, le=1, description="Intensidad de la cuantización (0–1)."
    )
    fill_gaps: float = Field(
        default=0.0, ge=0, le=16, description="Rellenar huecos menores (en subdivisiones)."
    )
    trim_leading_bars: bool = Field(
        default=False, description="Recortar compases iniciales vacíos."
    )


class ExportRequest(BaseModel):
    notes: list[ExportNote] = Field(description="Notas obtenidas del flujo SSE de /transcribe.")
    beat_grid: ExportGrid | None = Field(
        default=None, description="Rejilla de tempo devuelta en `transcription_complete`."
    )
    options: ExportSettings = Field(
        default_factory=ExportSettings,
        description="Ajustes de cuantización, tempo y compás.",
    )


def export_events(notes: list[ExportNote]) -> list[NoteStartEvent | NoteEndEvent]:
    """`notes` as the event stream `events_to_midi_bytes` reassembles them from."""
    events: list[NoteStartEvent | NoteEndEvent] = []
    for index, note in enumerate(notes):
        start = NoteStartEvent(
            pitch=note.pitch,
            start_time=note.start,
            index=index,
            instrument=note.instrument,
        )
        events += [start, NoteEndEvent(end_time=note.end, start_event=start)]
    return events


def event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def create_app(model: TranscriptionModel, web_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="MuScriptor API",
        description=_API_DESCRIPTION,
        version=version("muscriptor"),
        openapi_tags=_OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    transcribe_lock = threading.Lock()
    # State of the run currently holding the lock (or the last one to have held
    # it), guarded by `cancel_guard`: its cancel event and the id of the client
    # that started it. A new /transcribe from the SAME client (a resubmit in
    # the same browser tab) sets the cancel event so the in-flight run stops at
    # its next event boundary instead of transcribing to completion for a client
    # that has moved on. A request from a DIFFERENT client never preempts — it
    # waits for the lock like any other contender, so two independent browser
    # windows don't kill each other's transcription. This scoping must be done
    # here rather than by watching the socket: browser aborts don't always reach
    # us (e.g. port forwards / proxies keep the upstream connection open after
    # the browser aborts), so a same-tab resubmit can't be detected as a
    # disconnect.
    cancel_guard = threading.Lock()
    current_cancel: threading.Event | None = None
    current_client: str | None = None
    # Bound on how long a same-client resubmit waits for the run it just
    # cancelled to unwind and release the lock. That run stops within one
    # chunk boundary, so this only needs to cover that latency — unlike
    # genuine cross-client contention, which now fails instantly instead of
    # sitting on the connection (see below).
    preempt_wait_s = 5.0

    async def acquire_transcribe_lock(
        client_id: str | None, cancellable: bool
    ) -> tuple[threading.Event | None, Callable[[], None]]:
        """Acquire the single-transcription lock, preempting only a run started
        by this same `client_id` (a resubmit): that run is signalled to stop
        and this call waits up to `preempt_wait_s` for it to release the lock.
        A different client — or an anonymous API caller with no id — never
        preempts and is never preempted; it gets an immediate 503 instead of
        waiting, so a caller retrying against another machine (e.g. behind a
        load balancer like Traefik) doesn't have to hold the connection open.

        Returns `(cancel, release)`: `cancel` is the new run's cancel event
        (`None` when `cancellable` is False, e.g. the blocking /transcribe/midi
        render, which can't be stopped mid-flight); `release` frees the lock at
        most once, from whichever cleanup path runs first.
        """
        nonlocal current_cancel, current_client
        deadline: float | None = None
        while True:
            with cancel_guard:
                # Re-check each iteration so that even a same-client run which
                # became "current" while we were already waiting (a resubmit
                # that beat us to the lock) gets cancelled too — the newest
                # request from a given client always wins.
                preempting = (
                    current_cancel is not None
                    and client_id is not None
                    and current_client == client_id
                )
                if preempting:
                    current_cancel.set()
            if not preempting:
                acquired = await asyncio.to_thread(transcribe_lock.acquire, False)
                if not acquired:
                    raise HTTPException(
                        status_code=503,
                        detail="server busy: another transcription is in progress",
                    )
                break
            if deadline is None:
                deadline = time.monotonic() + preempt_wait_s
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail="server busy: another transcription is in progress",
                )
            acquired = await asyncio.to_thread(
                transcribe_lock.acquire, True, min(0.1, remaining)
            )
            if acquired:
                break
        cancel = threading.Event() if cancellable else None
        with cancel_guard:
            current_cancel = cancel
            current_client = client_id
        return cancel, _make_release_once(transcribe_lock)

    @app.get(
        "/health",
        tags=["sistema"],
        summary="Estado del servidor",
        description="Comprueba que el proceso HTTP responde correctamente.",
    )
    async def health():
        return {"status": "ok"}

    @app.get(
        "/instruments",
        tags=["sistema"],
        summary="Instrumentos admitidos",
        description=(
            "Lista canónica de nombres para el campo `instruments` de "
            "/transcribe y /transcribe/midi. Use el string exacto (p. ej. "
            "`acoustic_piano`, no «piano»)."
        ),
    )
    async def list_instruments():
        return {
            "instruments": sorted(MT3_FULL_PLUS_GROUP_NAMES.keys()),
            "usage": (
                "multipart: repita `instruments=<nombre>` por instrumento; "
                "omita el campo para detección automática."
            ),
        }

    @app.get(
        "/soundfonts/MuseScore_General.sf3",
        tags=["sistema"],
        summary="Soundfont para la interfaz web",
        description=(
            "Descarga (y cachea en disco) el soundfont comprimido usado por el "
            "sintetizador en el navegador."
        ),
    )
    async def soundfont() -> FileResponse:
        path = await asyncio.to_thread(download_if_necessary, SF3_URL)
        return FileResponse(path, media_type="application/octet-stream")

    @app.post(
        "/transcribe",
        tags=["transcripción"],
        summary="Transcribir audio (eventos SSE)",
        description=(
            "Sube un archivo de audio (`multipart/form-data`, campo **file**). "
            "Formatos habituales: WAV, MP3, FLAC, OGG, M4A (vía libsndfile).\n\n"
            "La respuesta es `text/event-stream`. Cada evento JSON incluye `type`:\n"
            "- `start` / `end`: notas (misma forma que en la CLI)\n"
            "- `progress`: avance por fragmentos `{completed, total}`\n"
            "- `transcription_complete`: MIDI en base64 (`data`) y `beat_grid` "
            "(tempo, compás, `onset_delay`, `steady`, lista `beats`, etc.)\n\n"
            "Solo puede ejecutarse una transcripción a la vez en el servidor. "
            "La cabecera opcional **X-Client-Id** permite que una nueva petición "
            "del mismo cliente cancele la anterior.\n\n"
            "Errores habituales: **400** audio ilegible o instrumento desconocido; "
            "**503** servidor ocupado.\n\n"
            "Véase el campo **instruments** para el listado de nombres válidos."
        ),
    )
    async def transcribe(
        file: Annotated[
            UploadFile,
            File(description="Archivo de audio a transcribir."),
        ],
        instruments: Annotated[
            list[InstrumentGroup],
            Form(
                description=_instruments_form_description(),
                default_factory=list,
            ),
        ],
        detect_tempo: Annotated[
            TempoDetection,
            Form(
                description=(
                    "Detección de tempo: `false` (no intentar), `true` (fallar si falla), "
                    "`best-effort` (intentar sin error)."
                ),
            ),
        ] = "best-effort",
        x_client_id: Annotated[
            str | None,
            Header(
                description=(
                    "Identificador de cliente (p. ej. pestaña del navegador). "
                    "Mismo id: la nueva transcripción cancela la anterior."
                ),
            ),
        ] = None,
    ) -> StreamingResponse:
        data = await file.read()
        # PCM WAV goes through the stdlib reader (keeps WAV decoding byte-for-byte
        # identical to the CLI); anything that isn't a readable WAV (mp3, flac,
        # ogg, m4a, …) falls back to soundfile/libsndfile. A genuinely
        # undecodable upload (corrupt/truncated file, or a format libsndfile
        # can't read) is the client's fault, so report it as a 400 rather than
        # letting it surface as a 500.
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        instrument_names = _instruments_for_model(instruments)

        # Acquire the single-transcription lock, preempting only a resubmit from
        # this same client (see acquire_transcribe_lock). `release_lock` runs
        # from whichever cleanup path fires first: the generator's finally
        # (normal completion, errors, mid-stream disconnects) or the
        # StreamingResponse background task (client disconnects before the
        # generator is ever iterated, so its finally would never run) — either
        # way the lock is released exactly once and never leaked.
        cancel, release_lock = await acquire_transcribe_lock(
            x_client_id, cancellable=True
        )

        def gen():
            try:
                events: list[NoteStartEvent | NoteEndEvent] = []
                # batch_size=1 so each chunk's notes stream out as soon as it is
                # generated, instead of waiting for a whole batch of chunks.
                # no_eos_is_ok=True so one runaway chunk that never emits EOS only
                # warns (and keeps its notes) instead of aborting the whole stream.
                for ev in model.transcribe(
                    (wav, sr),
                    instruments=instrument_names,
                    batch_size=1,
                    no_eos_is_ok=True,
                ):
                    # A newer request preempted this run — stop generating
                    # (closing the model.transcribe generator) and release the
                    # lock via the finally, at most one chunk after the signal.
                    if cancel.is_set():
                        return
                    if isinstance(ev, ProgressEvent):
                        # Coarse chunk-completion anchor — forward it but keep it
                        # out of the note list the MIDI file is built from.
                        payload = json.dumps(
                            {
                                "type": "progress",
                                "completed": ev.completed,
                                "total": ev.total,
                            }
                        )
                        yield f"data: {payload}\n\n"
                        continue
                    events.append(ev)
                    payload = json.dumps(event_to_dict(ev))
                    yield f"data: {payload}\n\n"
                # All notes streamed — build the MIDI file in memory (reusing the
                # exact `muscriptor transcribe` logic) and send it as a final event
                # with the bytes base64-encoded.
                if cancel.is_set():
                    return
                # Detect tempo/meter only now: it costs a few seconds of CPU and
                # nothing before this point needs it, so the notes stream first.
                grid = model.detect_beat_grid_for((wav, sr), detect_tempo)
                # Measure the onset lag up here rather than leaving it to the MIDI
                # writing, since the UI has to be told the very same number to move
                # the notes it already drew.
                if grid:
                    grid = grid.with_onset_delay(
                        [
                            ev.start_time
                            for ev in events
                            if isinstance(ev, NoteStartEvent)
                        ]
                    )
                midi_bytes = model.events_to_midi_bytes(iter(events), beat_grid=grid)
                midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
                # The grid rides along so the UI can draw bar lines instead of a
                # fixed seconds grid; null when no tempo was detected.
                payload = json.dumps(
                    {
                        "type": "transcription_complete",
                        "data": midi_b64,
                        # Only the fields the UI draws with; `grid.beats` is an
                        # ndarray and not JSON-serializable anyway.
                        "beat_grid": {
                            "bpm": grid.bpm,
                            "beats_per_bar": grid.beats_per_bar,
                            "first_downbeat": grid.first_downbeat,
                            # Seconds the streamed note times sit late against
                            # the beats; the MIDI already has it taken out, the
                            # UI has to subtract it from the notes it drew.
                            "onset_delay": grid.onset_delay,
                            # What /export quantizes to unless told otherwise.
                            "beat_subdivision": grid.beat_subdivision,
                            "steady": grid.steady,
                            # For the UI to draw bar lines that follow them, and
                            # to send back to /export.
                            "beats": None
                            if grid.beats is None
                            else [round(float(b), 3) for b in grid.beats],
                        }
                        if grid
                        else None,
                    }
                )
                yield f"data: {payload}\n\n"
            finally:
                release_lock()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            background=BackgroundTask(release_lock),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/transcribe/midi",
        tags=["transcripción"],
        summary="Transcribir audio (archivo MIDI)",
        description=(
            "Igual que /transcribe en cuanto a entrada (audio + instrumentos + "
            "`detect_tempo`), pero **bloquea** hasta terminar y devuelve "
            "`audio/midi` con cabecera de descarga (sin SSE ni base64).\n\n"
            "Límite de duración: **15 minutos** (**413** si se supera). "
            "**422** si `detect_tempo=true` y falla la detección de tempo. "
            "**503** si otra transcripción está en curso."
        ),
    )
    async def transcribe_midi(
        file: Annotated[
            UploadFile,
            File(description="Archivo de audio a transcribir."),
        ],
        instruments: Annotated[
            list[InstrumentGroup],
            Form(
                description=_instruments_form_description(),
                default_factory=list,
            ),
        ],
        detect_tempo: Annotated[
            TempoDetection,
            Form(description="Modo de detección de tempo (véase /transcribe)."),
        ] = "best-effort",
        x_client_id: Annotated[
            str | None,
            Header(description="Identificador de cliente (véase /transcribe)."),
        ] = None,
    ) -> Response:
        data = await file.read()
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except (wave.Error, EOFError):
            try:
                wav, sr = _read_non_wav_file(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not decode audio file '{file.filename}': {e}",
                ) from e

        duration_s = wav.shape[-1] / sr
        if duration_s > _MAX_TRANSCRIBE_MIDI_DURATION_S:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio file is {duration_s / 60:.1f} minutes long; "
                    f"the limit is {_MAX_TRANSCRIBE_MIDI_DURATION_S // 60:.0f} minutes"
                ),
            )

        instrument_names = _instruments_for_model(instruments)

        # Same mutual exclusion as /transcribe, via the shared helper. This run
        # is not cancellable (it doesn't stream, so there's nothing to stop
        # mid-flight); cancellable=False records that, so nothing tries to
        # preempt a blocking MIDI render. A non-browser API caller sends no
        # client id and so neither preempts nor is preempted — it just waits for
        # the lock like any other contender.
        _cancel, release_lock = await acquire_transcribe_lock(
            x_client_id, cancellable=False
        )
        try:
            midi_bytes, _ = await asyncio.to_thread(
                model.transcribe_and_postprocess,
                (wav, sr),
                instruments=instrument_names,
                detect_tempo=detect_tempo,
            )
        except BeatDetectionError as e:
            # Only reachable with detect_tempo=true, where the caller wants an error
            # if tempo detection fails.
            raise HTTPException(status_code=422, detail=str(e)) from e
        finally:
            release_lock()

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="result.mid"'},
        )

    @app.post(
        "/export",
        tags=["exportación"],
        summary="Exportar MIDI desde notas",
        description=(
            "Recibe en JSON las notas del SSE, la `beat_grid` opcional y "
            "`options` (tempo, compás, cuantización, etc.). Devuelve un archivo "
            "**MIDI** (`audio/midi`). No ejecuta el modelo ni usa el candado de "
            "/transcribe, así que es barato para reexportar al cambiar opciones."
        ),
    )
    async def export(request: ExportRequest) -> Response:
        unknown = sorted(
            {n.instrument for n in request.notes} - MT3_FULL_PLUS_GROUP_NAMES.keys()
        )
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )
        grid = None
        if request.beat_grid is not None:
            fields = request.beat_grid.model_dump()
            beats = fields.pop("beats")
            grid = BeatGrid(
                **fields, beats=None if beats is None else np.asarray(beats)
            )
        midi_bytes = await asyncio.to_thread(
            model.events_to_midi_bytes,
            iter(export_events(request.notes)),
            beat_grid=grid,
            export=ExportOptions(**request.options.model_dump()),
        )
        return Response(content=midi_bytes, media_type="audio/midi")

    @app.post(
        "/auralize",
        tags=["audio"],
        summary="Preescuchar transcripción (WAV)",
        description=(
            "Renderiza el MIDI subido a audio WAV.\n\n"
            "- **mix**: estéreo — audio original (izquierda) + síntesis FluidSynth "
            "(derecha); requiere también el campo **audio**.\n"
            "- **synth**: mono, solo la síntesis."
        ),
    )
    async def auralize(
        midi: Annotated[UploadFile, File(description="Archivo MIDI transcrito.")],
        audio: Annotated[
            UploadFile | None,
            File(description="Audio original; obligatorio si `mode=mix`."),
        ] = None,
        mode: Annotated[
            str,
            Form(description="`mix` (mezcla con el original) o `synth` (solo síntesis)."),
        ] = "mix",
    ):
        from muscriptor.utils.auralization import auralize as do_auralize
        from muscriptor.utils.auralization import synthesize

        if mode not in ("mix", "synth"):
            raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")
        if mode == "mix" and audio is None:
            raise HTTPException(
                status_code=400, detail="mode='mix' requires an audio file"
            )

        midi_data = await midi.read()
        tmp_paths: list[str] = []

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            tmp_midi.write(midi_data)
            midi_tmp = tmp_midi.name
            tmp_paths.append(midi_tmp)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_tmp = tmp_out.name
            tmp_paths.append(out_tmp)

        try:
            if mode == "synth":
                synthesize(midi_path=midi_tmp, output_path=out_tmp)
            else:
                audio_data = await audio.read()
                suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp_audio:
                    tmp_audio.write(audio_data)
                    tmp_paths.append(tmp_audio.name)
                do_auralize(
                    midi_path=midi_tmp,
                    original_audio_path=tmp_audio.name,
                    output_path=out_tmp,
                )
            with open(out_tmp, "rb") as f:
                wav_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)

        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post(
        "/sheets",
        tags=["exportación"],
        summary="Partituras en ZIP",
        description=(
            "Sube un MIDI (idealmente cuantizado desde /export con "
            "`quantize=true`) y devuelve un **ZIP** sin comprimir con MusicXML, "
            "partitura completa y un PDF por instrumento.\n\n"
            "Requiere **MuseScore 4+** instalado en el servidor (**503** si no "
            "está disponible). El campo **quantized** indica que el MIDI ya está "
            "alineado a la rejilla; este endpoint no cuantiza por sí mismo."
        ),
    )
    async def sheets(
        midi: Annotated[UploadFile, File(description="Archivo MIDI a grabar.")],
        quantized: Annotated[
            bool,
            Form(
                description=(
                    "True si el MIDI proviene de /export con cuantización "
                    "(recomendado para la notación)."
                ),
            ),
        ] = False,
    ) -> Response:
        midi_bytes = await midi.read()

        try:
            zip_bytes = await asyncio.to_thread(engrave_to_zip, midi_bytes, quantized)
        except MuseScoreNotFoundError as e:
            # A deployment problem, not a bad request: the same 503 the UI
            # already knows how to report, with the install hint as its detail.
            raise HTTPException(status_code=503, detail=str(e)) from e
        except MuseScoreError as e:
            # MuseScore ran but wrote nothing usable — most often because the
            # upload wasn't a MIDI file it could import.
            raise HTTPException(status_code=500, detail=str(e)) from e

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{SHEETS_ZIP_NAME}"'
            },
        )

    if web_dir is not None:
        web_path = Path(web_dir)
        if web_path.is_dir():
            app.mount("/", StaticFiles(directory=web_path, html=True), name="web")

    return app
