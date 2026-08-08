# Transcription pipeline

Speech-to-text service: accepts an audio or video file in any format ffmpeg can decode,
returns the transcript plus per-segment timestamps, and handles files long enough that a
single HTTP request would time out.

Runs entirely locally on [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper via CTranslate2). No API key, no per-minute cost, no audio leaving the machine.

## Files

| File | What it is |
| --- | --- |
| `transcriber.py` | The pipeline: format probing and normalisation, transcription, segment timestamps, long-file windowing, SRT/VTT output. Runnable as a CLI. |
| `api.py` | FastAPI service: file upload, inline transcription, background jobs with progress. |
| `index.html` | Minimal upload page, served by the API at `/`. |
| `ANSWERS.md` | Written answers to the assignment questions, including the bugs found while building it. |

## Setup

Needs `ffmpeg` and `ffprobe` on `PATH`.

```bash
brew install ffmpeg          # or: apt install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# logic self-check — no model or audio needed
python transcriber.py --self-check

# CLI
python transcriber.py harvard.wav              # segments + full text
python transcriber.py harvard.wav --json       # full result as JSON
python transcriber.py harvard.wav --srt        # subtitles
python transcriber.py harvard.wav --words      # word-level timestamps

# service — then open http://127.0.0.1:8000
uvicorn api:app --reload
```

```bash
curl -F "file=@harvard.wav" localhost:8000/transcribe          # short files, inline
curl -F "file=@long-recording.mp3" localhost:8000/jobs         # long files -> 202 + job id
curl localhost:8000/jobs/<id>                                  # status + progress + result
curl localhost:8000/jobs/<id>/srt                              # subtitles
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Upload page |
| `GET` | `/health` | Liveness + which model is loaded |
| `POST` | `/transcribe` | Transcribe now, returns text + segments. `413` if the audio is too long — use `/jobs`. |
| `POST` | `/jobs` | Any length. `202` with a job id, work runs in the background. |
| `GET` | `/jobs/{id}` | `queued` / `processing` / `done` / `failed`, plus progress and result |
| `GET` | `/jobs/{id}/srt` | SRT subtitles for a finished job |

## Response shape

`text` is the plain transcript; `segments` carries the timestamps. Same pass, one code path.

```json
{
  "language": "en",
  "duration": 18.356,
  "chunks": 1,
  "segments": [
    {"id": 0, "start": 0.91, "end": 3.91, "text": "The stale smell of old beer lingers."},
    {"id": 1, "start": 3.91, "end": 6.91, "text": "It takes heat to bring out the odor."}
  ],
  "text": "The stale smell of old beer lingers. It takes heat to bring out the odor.",
  "source": {"container": "wav", "codec": "pcm_s16le", "channels": 1,
             "sample_rate": 16000, "duration": 18.356}
}
```

## Configuration

Environment variables, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WHISPER_MODEL` | `small` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `int8` | Quantisation, e.g. `int8`, `float16` |
| `CHUNK_SECONDS` | `600` | Target window length for long files |
| `MAX_DURATION_SECONDS` | `14400` | Reject anything longer, up front |

## How long files are handled

Long audio has two unrelated problems, so it gets two answers.

**The request would time out.** Anything long goes through `POST /jobs`, which validates the
upload, returns a job id immediately, and transcribes in the background while `GET /jobs/{id}`
reports progress.

**Memory and accuracy.** The file is never fully decoded into memory — two hours at 16 kHz is
roughly half a gigabyte of samples. It is walked in ~10-minute windows cut straight from disk,
each window's temp file deleted as soon as it is transcribed, so peak memory and disk stay flat
no matter how long the input is.

Three details make the windowing lossless, and each is a silent bug if skipped:

1. **Cut on silence, not on a fixed clock.** `silencedetect` finds the gaps and each boundary
   snaps to the nearest one. A blind cut at exactly ten minutes lands mid-word.
2. **Advance to the last complete segment, not to the cut point.** After each window the final
   segment (usually clipped by the cut) is discarded and the next window resumes where the last
   complete segment ended. This removes the need for a fixed overlap plus deduplication, which
   was the first approach and was abandoned: segment boundaries never align to cut points, so
   any drop rule either loses a sentence or duplicates one.
3. **Clamp each window's timestamps to its real length.** Whisper pads audio to 30 seconds
   internally and will report timestamps past the end of a short clip. A 7.06 s window returned
   a segment ending at 8.20 s; trusting that pushed the cursor past 1.1 s of speech that was
   never transcribed, and those words disappeared with no error.

Verified by transcribing the same file with `CHUNK_SECONDS` forced to 6, 8 and 30 and confirming
all three produce output identical to the single-pass transcript. That is the check worth
writing, because a chunking bug does not crash — it quietly drops words.

## Audio formats

Formats are not handled in Python. Everything is normalised at the edge so exactly one format
reaches the model:

1. The upload is streamed to a temp file under a size cap, never read fully into memory.
2. `ffprobe` reports the real container, codec, channels, sample rate and duration. Validation
   uses that, **never the file extension** — a file named `.mp3` can contain anything. No audio
   stream, or ffprobe failing, gives `415`.
3. `ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le` converts to 16 kHz mono PCM WAV, what Whisper
   wants internally. `-vn` drops video, so `.mp4`/`.mov` recordings work off their audio track.
4. Temp files are removed in a `finally` block.

So the supported list is "whatever ffmpeg decodes" — mp3, m4a/aac, wav, flac, ogg, opus, wma,
plus video containers — rather than a hardcoded enum. ffmpeg is always invoked with an argument
list, never `shell=True`, on a server-generated path, so a hostile filename cannot reach a shell.

## Known limits

- The job store is an in-process dict. Fine for one worker; becomes Redis plus Celery/RQ the
  moment you need several workers or want jobs to survive a restart.
- Windows are transcribed sequentially. Parallel decoding only helps with spare CPU/GPU and
  complicates progress reporting.
- No checkpoint/resume for a job interrupted mid-file.

`harvard.wav` is a recording of the Harvard sentences, used as the test sample.
