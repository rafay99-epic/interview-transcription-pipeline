"""Speech-to-text core: format normalisation, transcription, segment timestamps,
long-file chunking.

Answers questions 1-4 of the assignment with a single code path:
  Q1 transcribe()["text"]      -> plain text
  Q2 transcribe()["segments"]  -> per-segment timestamps
  Q3 probe() + to_wav16k()     -> any ffmpeg-decodable format
  Q4 plan_chunks() + merge_segments() -> long audio without blowing up memory
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

# Files longer than this get split. ~10 min keeps peak RAM small and lets us
# report progress; short files take the single-pass path.
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "600"))
# How far from the ideal cut point we will search for a silence to cut on.
SILENCE_SEARCH_WINDOW = 60.0

MAX_DURATION_SECONDS = float(os.getenv("MAX_DURATION_SECONDS", "14400"))  # 4 h

_model: WhisperModel | None = None


def model() -> WhisperModel:
    """Load once per process. Model load costs seconds; do not do it per request."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


# --------------------------------------------------------------------------
# Q3: audio formats
# --------------------------------------------------------------------------

def probe(path: str | Path) -> dict:
    """Read real container/codec info. Trust the bytes, never the extension."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate:format=duration,format_name",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise ValueError(f"not a decodable media file: {proc.stderr.strip().splitlines()[-1:]}")

    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError("file has no audio stream")

    stream = streams[0]
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError("could not determine duration")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(f"audio too long: {duration:.0f}s > {MAX_DURATION_SECONDS:.0f}s limit")

    return {
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "channels": int(stream.get("channels") or 0),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "duration": duration,
    }


def to_wav16k(src: str | Path, dst: str | Path) -> None:
    """Normalise anything ffmpeg can read to 16 kHz mono PCM, what Whisper wants.

    Everything downstream sees exactly one format, so mp3/m4a/ogg/flac/opus/mp4
    stop being our problem and become ffmpeg's.
    """
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-i", str(src),
            "-vn",                     # drop video stream (mp4/mov inputs)
            "-ac", "1",                # mono
            "-ar", "16000",            # 16 kHz
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True, capture_output=True, timeout=3600,
    )


# --------------------------------------------------------------------------
# Q4: long audio
# --------------------------------------------------------------------------

def silence_points(path: str | Path, noise: str = "-30dB", min_dur: float = 0.5) -> list[float]:
    """Timestamps where a silent stretch ends. Used as candidate cut points."""
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "info", "-i", str(path),
            "-af", f"silencedetect=noise={noise}:d={min_dur}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=3600,
    )
    return [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", proc.stderr)]


def next_end(
    cursor: float,
    duration: float,
    silences: Iterable[float],
    target: float = CHUNK_SECONDS,
    window: float = SILENCE_SEARCH_WINDOW,
) -> float:
    """Where the chunk starting at `cursor` should stop.

    Prefers a silence near the ideal boundary so a cut never lands mid-word.
    Pure function: testable without touching audio.
    """
    if duration - cursor <= target * 1.2:
        return duration                       # avoid leaving a runt final chunk
    ideal = cursor + target
    floor = cursor + min(30.0, target * 0.5)
    near = [s for s in silences if abs(s - ideal) <= window and s > floor]
    return min(near, key=lambda s: abs(s - ideal)) if near else ideal


def cut(src: str | Path, start: float, end: float, dst: str | Path) -> None:
    """Extract [start, end) to its own wav. Streams from disk, so a 4-hour file
    never sits in RAM."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.01):.3f}",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True, capture_output=True, timeout=3600,
    )


def merge_segments(acc: list[dict], segs: Iterable[dict], offset: float) -> list[dict]:
    """Shift a chunk's segments back onto the global timeline and append them.

    Every chunk's decoder output starts at 0.0. Forgetting this offset is the
    classic long-audio bug: chunk 2 claims to start at second zero.
    """
    for seg in segs:
        start = seg["start"] + offset
        end = seg["end"] + offset
        text = seg["text"].strip()
        if not text:
            continue
        # Drop a segment only when its MIDPOINT falls inside what we already
        # have. Testing `start` instead throws away the boundary segment, which
        # normally begins inside the overlap but carries mostly new speech.
        if acc and (start + end) / 2 < acc[-1]["end"]:
            continue
        item = {
            "id": len(acc),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        }
        if seg.get("words"):
            item["words"] = [
                {
                    "word": w["word"],
                    "start": round(w["start"] + offset, 3),
                    "end": round(w["end"] + offset, 3),
                    "probability": round(w.get("probability", 0.0), 4),
                }
                for w in seg["words"]
            ]
        acc.append(item)
    return acc


def clamp_segments(segs: list[dict], limit: float) -> list[dict]:
    """Whisper pads audio to 30 s internally and will happily report timestamps
    past the end of a short chunk. Left unclamped, that overshoot pushes the
    cursor beyond audio we never actually transcribed, silently skipping speech.
    """
    out = []
    for seg in segs:
        seg = dict(seg)
        seg["start"] = min(seg["start"], limit)
        seg["end"] = min(seg["end"], limit)
        if seg["end"] - seg["start"] < 0.05:
            continue
        if seg.get("words"):
            seg["words"] = [
                {**w, "start": min(w["start"], limit), "end": min(w["end"], limit)}
                for w in seg["words"]
                if w["start"] < limit
            ]
        out.append(seg)
    return out


def _as_dicts(segments) -> list[dict]:
    """faster-whisper yields lazily; force it and flatten to plain dicts so the
    merge logic stays testable."""
    out = []
    for s in segments:
        out.append(
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (s.words or [])
                ],
            }
        )
    return out


# --------------------------------------------------------------------------
# Q1 + Q2: one call, two views of the same result
# --------------------------------------------------------------------------

def transcribe(
    path: str | Path,
    word_timestamps: bool = False,
    language: str | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Transcribe any audio/video file.

    Returns {"text", "segments", "language", "duration", "source", "chunks"}.
    `text` answers Q1, `segments` answers Q2 - same pass, no second engine.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    meta = probe(path)

    with tempfile.TemporaryDirectory(prefix="stt-") as tmpdir:
        tmp = Path(tmpdir)
        wav = tmp / "audio.wav"
        to_wav16k(path, wav)

        duration = meta["duration"]
        silences = sorted(silence_points(wav)) if duration > CHUNK_SECONDS else []

        segments: list[dict] = []
        detected = language
        cursor = 0.0
        index = 0

        # Walk the file window by window. The cursor advances to the end of the
        # last COMPLETE segment rather than to the cut point, so the next window
        # re-decodes whatever the cut interrupted. That is why no fixed overlap
        # is needed and why nothing is lost at a boundary.
        while cursor < duration - 0.5:
            end = next_end(cursor, duration, silences)
            final = end >= duration

            if cursor == 0.0 and final:
                piece = wav                       # short file: no cutting at all
            else:
                piece = tmp / f"chunk_{index:04d}.wav"
                cut(wav, cursor, end, piece)

            raw, info = model().transcribe(
                str(piece),
                language=detected,          # lock language after the first window
                vad_filter=True,            # skip silence, avoids hallucinated text
                word_timestamps=word_timestamps,
                beam_size=5,
            )
            segs = clamp_segments(_as_dicts(raw), end - cursor)
            detected = detected or info.language

            # The last segment of a non-final window is usually clipped by the
            # cut; drop it and let the next window transcribe it in full.
            if not final and len(segs) > 1:
                segs = segs[:-1]

            before = len(segments)
            merge_segments(segments, segs, cursor)
            index += 1

            if piece != wav:
                piece.unlink(missing_ok=True)     # keep disk use flat

            if final:
                cursor = duration
            elif len(segments) > before:
                cursor = max(segments[-1]["end"], cursor + 1.0)  # +1.0 forces progress
            else:
                cursor = end                      # window was pure silence

            if progress:
                progress(min(cursor / duration, 1.0))

    return {
        "language": detected,
        "duration": round(duration, 3),
        "chunks": index,
        "segments": segments,
        "text": " ".join(s["text"] for s in segments).strip(),
        "source": meta,
    }


# --------------------------------------------------------------------------
# Subtitle output - segments are already in the right shape for it
# --------------------------------------------------------------------------

def _ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    return "\n".join(
        f"{i + 1}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(segments)
    )


def to_vtt(segments: list[dict]) -> str:
    body = "\n".join(
        f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n"
        for s in segments
    )
    return "WEBVTT\n\n" + body


# --------------------------------------------------------------------------
# Self-check + CLI
# --------------------------------------------------------------------------

def _self_check() -> None:
    # short file: one window covering everything
    assert next_end(0.0, 120.0, [], target=600.0) == 120.0

    # long file, no usable silence: falls back to the fixed window
    assert next_end(0.0, 1500.0, [], target=600.0) == 600.0
    assert next_end(600.0, 1500.0, [], target=600.0) == 1200.0
    assert next_end(1200.0, 1500.0, [], target=600.0) == 1500.0

    # a silence near the ideal boundary wins over the blind cut
    assert next_end(0.0, 1000.0, [595.0, 610.0, 900.0], target=600.0) == 595.0

    # a silence too close to the cursor is ignored (no runt chunks)
    assert next_end(0.0, 1000.0, [10.0], target=600.0) == 600.0

    # offsets are applied and overlap duplicates are dropped
    acc = merge_segments([], [{"start": 0.0, "end": 5.0, "text": "hello"}], 0.0)
    acc = merge_segments(
        acc,
        [
            {"start": 0.0, "end": 2.0, "text": "hello"},   # overlap repeat -> dropped
            {"start": 2.0, "end": 6.0, "text": "world"},
        ],
        3.0,
    )
    assert [s["text"] for s in acc] == ["hello", "world"], acc
    assert acc[1]["start"] == 5.0 and acc[1]["end"] == 9.0, acc
    assert [s["id"] for s in acc] == [0, 1]

    # a segment that starts inside the overlap but mostly covers new speech is
    # kept, not silently dropped
    acc = merge_segments([], [{"start": 0.0, "end": 8.0, "text": "first"}], 0.0)
    acc = merge_segments(acc, [{"start": 0.0, "end": 7.0, "text": "second"}], 6.0)
    assert [s["text"] for s in acc] == ["first", "second"], acc

    # timestamps that overshoot the chunk are pulled back, not trusted
    clamped = clamp_segments(
        [{"start": 0.5, "end": 8.2, "text": "x"}, {"start": 8.2, "end": 9.0, "text": "y"}],
        7.0,
    )
    assert [(s["start"], s["end"]) for s in clamped] == [(0.5, 7.0)], clamped

    assert _ts(3661.5) == "01:01:01,500"
    print("self-check OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--self-check":
        _self_check()
        sys.exit(0)

    result = transcribe(args[0], word_timestamps="--words" in args)

    if "--json" in args:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif "--srt" in args:
        print(to_srt(result["segments"]))
    else:
        print(f"\nlanguage={result['language']}  duration={result['duration']}s  "
              f"chunks={result['chunks']}\n")
        for s in result["segments"]:
            print(f"[{s['start']:8.2f} -> {s['end']:8.2f}] {s['text']}")
        print(f"\nFull text:\n{result['text']}")
