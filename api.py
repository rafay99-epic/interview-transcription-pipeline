"""HTTP service around transcriber.py.

  GET  /                  minimal browser upload page
  POST /transcribe        short files, transcribes inline, returns the result
  POST /jobs              any length, returns 202 + job id, work runs in background
  GET  /jobs/{id}         status / progress / result
  GET  /jobs/{id}/srt     subtitles for a finished job

Long audio never blocks a request: that is the whole point of the /jobs pair.
Run: uvicorn api:app --reload
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

import transcriber

app = FastAPI(title="Transcription service")

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
INLINE_LIMIT_SECONDS = 120.0  # longer than this, use /jobs

# ponytail: in-process job store, dies with the worker. Swap for Redis/Celery
# the moment you need more than one worker or restart-survival.
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def _save_upload(upload: UploadFile) -> Path:
    """Stream to disk with a size cap. Never read an upload fully into memory."""
    suffix = Path(upload.filename or "audio").suffix[:10]
    fd = tempfile.NamedTemporaryFile(prefix="upload-", suffix=suffix, delete=False)
    written = 0
    try:
        with fd:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large")
                fd.write(chunk)
    except Exception:
        Path(fd.name).unlink(missing_ok=True)
        raise
    return Path(fd.name)


@app.get("/")
def index() -> FileResponse:
    """Minimal upload page. Served from the API so there is no CORS to configure."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": transcriber.MODEL_SIZE}


@app.post("/transcribe")
def transcribe_now(file: UploadFile = File(...), words: bool = False) -> dict:
    """Q1 + Q2. Response `text` is the plain transcript, `segments` carries the
    per-segment timestamps."""
    path = _save_upload(file)
    try:
        meta = transcriber.probe(path)          # Q3: validate before decoding
        if meta["duration"] > INLINE_LIMIT_SECONDS:
            raise HTTPException(
                413,
                f"{meta['duration']:.0f}s audio: POST /jobs instead of /transcribe",
            )
        return transcriber.transcribe(path, word_timestamps=words)
    except ValueError as e:
        raise HTTPException(415, str(e))        # unreadable / no audio stream
    finally:
        path.unlink(missing_ok=True)


def _run_job(job_id: str, path: Path, words: bool) -> None:
    def progress(fraction: float) -> None:
        with _lock:
            JOBS[job_id]["progress"] = round(fraction, 3)

    try:
        with _lock:
            JOBS[job_id]["status"] = "processing"
        result = transcriber.transcribe(path, word_timestamps=words, progress=progress)
        with _lock:
            JOBS[job_id].update(status="done", progress=1.0, result=result)
    except Exception as e:
        with _lock:
            JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        path.unlink(missing_ok=True)


@app.post("/jobs", status_code=202)
def create_job(background: BackgroundTasks, file: UploadFile = File(...), words: bool = False) -> dict:
    """Q4. Accept the file, hand back an id immediately, transcribe out of band.
    A two-hour recording would time out any sane HTTP client otherwise."""
    path = _save_upload(file)
    try:
        meta = transcriber.probe(path)
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(415, str(e))

    job_id = uuid.uuid4().hex
    with _lock:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "filename": file.filename,
            "duration": meta["duration"],
        }
    background.add_task(_run_job, job_id, path, words)
    return {"id": job_id, "status": "queued", "duration": meta["duration"],
            "poll": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/jobs/{job_id}/srt", response_class=PlainTextResponse)
def job_srt(job_id: str) -> str:
    with _lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return transcriber.to_srt(job["result"]["segments"])
