# Answers

Code: https://github.com/rafay99-epic/interview-transcription-pipeline

Run `python transcriber.py --self-check` to exercise the pipeline logic without needing a
model or an audio file.

---

## 1. Implement a service or script that accepts an audio file (e.g., WAV or MP3)

There are two entry points into the same core: a CLI (`python transcriber.py recording.mp3`)
and an HTTP endpoint (`POST /transcribe`, multipart upload), plus a small browser page served
at `/` so it can be tried without curl.

The part I was careful about is the upload path. The file is streamed to a temp file in 1 MB
chunks under a size cap rather than read into memory, because reading whole uploads into RAM is
how a service falls over as soon as two people upload an hour of audio at once. Once it is on
disk, `ffprobe` checks it is genuinely decodable media with an audio stream before anything
tries to transcribe it, so a corrupt or mislabelled file fails immediately with a `415` instead
of surfacing as a confusing decoder error later. Oversized uploads get a `413`. Temp files are
removed in a `finally` block so a failure mid-request does not leave them behind.

WAV and MP3 both work, and so does anything else ffmpeg can read — see the formats answer for
why that is not a hardcoded list.

Code: `api.py` (`_save_upload`, `POST /transcribe`) and `transcriber.py`.

---

## 2. Implement a service or script that transcribes spoken language into text

I used faster-whisper, which is Whisper reimplemented on CTranslate2, running the `small` model
locally with int8 quantisation on CPU. I chose local inference over a hosted API deliberately:
there is no API key to manage, no per-minute cost on long recordings, no network dependency in
the request path, and no customer audio leaving the machine. `transcriber.transcribe(path)`
returns a result whose `text` field is the transcript, and `POST /transcribe` exposes the same
thing over HTTP.

Two details that turned out to matter. The model is loaded once per process and cached, because
loading it costs several seconds and doing that per request would dominate the latency of every
call. And faster-whisper returns a lazy generator of segments — nothing is actually decoded
until you iterate it, so the work has to be forced before you can count segments, time the run,
or trust the detected language. It is an easy thing to get wrong and it makes the code look
misleadingly fast.

The tradeoff I accepted is accuracy: a hosted model like Deepgram or `whisper-1` would do better
on noisy or accented audio. Because everything downstream works on a normalised list of
segments rather than on the engine's own types, switching backends means replacing the body of
one function.

Code: https://github.com/rafay99-epic/interview-transcription-pipeline — `transcriber.py`

---

## 3. Implement a service or script that returns the transcription with timestamps per segment

Same engine call and same pass — I did not write a second code path for this. Whisper already
produces segments carrying `start` and `end`, so the plain transcript from the previous question
is just those segments joined. One request returns both views:

```json
{
  "language": "en",
  "duration": 18.356,
  "chunks": 1,
  "segments": [
    {"id": 0, "start": 0.91, "end": 3.91, "text": "The stale smell of old beer lingers."},
    {"id": 1, "start": 3.91, "end": 6.91, "text": "It takes heat to bring out the odor."}
  ],
  "text": "The stale smell of old beer lingers. It takes heat to bring out the odor."
}
```

Word-level timestamps are available behind a `words=true` flag rather than on by default,
because they slow decoding noticeably and most callers only need segments.

I also enable Whisper's VAD filter, which skips silent regions. That is partly speed, but mainly
correctness: fed silence, Whisper tends to invent text, and filtering it out avoids transcripts
with confident-sounding sentences that nobody said.

Because the segments already carry timing, subtitle output is nearly free — `to_srt()` and
`to_vtt()` format the same list, and `GET /jobs/{id}/srt` returns a ready SRT file.

Code: https://github.com/rafay99-epic/interview-transcription-pipeline — `transcriber.py`

---

## 4. How do you handle different audio formats?

I don't handle formats in Python at all — I normalise everything with ffmpeg at the edge so that
exactly one format ever reaches the model.

The upload is streamed to a temp file, then `ffprobe` reports the real container, codec, channel
count, sample rate and duration, and validation uses that rather than the file extension. That
distinction matters: a file named `.mp3` can contain anything, so trusting the extension means
trusting the client. If there is no audio stream, or ffprobe cannot read the file at all, it is
rejected with a `415` before any decoding is attempted. Then
`ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le` converts it to 16 kHz mono PCM WAV, which is what
Whisper wants internally anyway. The `-vn` flag drops the video stream, so `.mp4` and `.mov`
recordings work off their audio track. Temp files are cleaned up in a `finally` block.

The practical benefit is that the supported-format list becomes "whatever ffmpeg can decode" —
mp3, m4a/aac, wav, flac, ogg, opus, wma, plus video containers — instead of a hardcoded enum I
would have to keep extending every time someone uploads something new. I verified WAV, MP3 and
M4A inputs produce identical transcripts.

On safety, ffmpeg is always invoked with an argument list rather than `shell=True`, and always on
a path the server generated, so a hostile filename cannot reach a shell.

Code: https://github.com/rafay99-epic/interview-transcription-pipeline — `transcriber.py`
(`probe`, `to_wav16k`)

---

## 5. How do you deal with long audio files?

Long audio has two unrelated problems, and I think they need two separate answers.

The first is the request lifetime. A two-hour recording takes minutes to transcribe and no HTTP
client will wait for that, so long files go through a job API instead. `POST /jobs` validates the
upload, returns `202` with a job id straight away, and does the work in the background;
`GET /jobs/{id}` reports `queued`, `processing`, `done` or `failed`, along with a progress
fraction and the result when it is ready. The job store is an in-process dict, which is honest
for a single worker and would become Redis plus a Celery or RQ queue the moment you need several
workers or want jobs to survive a restart.

The second is memory and accuracy. I never decode the whole file into memory — two hours at
16 kHz is roughly half a gigabyte of samples. Instead the file is walked in roughly ten-minute
windows cut straight from disk with ffmpeg, and each window's temp file is deleted as soon as it
has been transcribed, so peak memory and disk stay flat no matter how long the input is.

Getting that windowing to be lossless took three specific things, and each of them is a silent
bug if you skip it. First, cut on silence rather than on a fixed clock: I run ffmpeg's
`silencedetect` first and snap each boundary to the nearest silence near the ideal cut point,
because a blind cut at exactly ten minutes lands mid-word. Second, advance the cursor to the end
of the last complete segment rather than to the cut point — after each window I discard its final
segment, which the cut usually clipped, and let the next window transcribe it in full. That
removes the need for a fixed overlap plus deduplication, which was my first approach and which I
abandoned, because Whisper's segment boundaries never align with your cut points, so any dedup
rule either loses a sentence or duplicates one. Third, clamp each window's timestamps to the
window's real length. Whisper pads audio to 30 seconds internally and will happily report
timestamps past the end of a short clip; in my testing a 7.06-second window returned a segment
ending at 8.20 seconds, and trusting that pushed the cursor past 1.1 seconds of speech that was
never transcribed. Those words just disappeared, with no error anywhere.

Every window's timestamps are then shifted by that window's offset before being merged onto the
global timeline — forgetting that offset is the classic version of this bug, where the second
chunk claims to start at second zero.

I verified all of this by transcribing the same file with the window size forced to 6, 8 and 30
seconds and confirming all three produce output identical to the single-pass transcript. That is
the check worth writing, because a chunking bug does not crash. It quietly drops words, and you
only notice when a customer reads the transcript.

Also in place: a maximum-duration guard evaluated from the ffprobe duration, so an absurd upload
is rejected in milliseconds rather than after an hour of work, and per-job error isolation so one
bad file cannot take the worker down. Deliberately not built: parallel window decoding, which
only helps if there is spare CPU or GPU and complicates progress reporting, and
checkpoint/resume for a job interrupted mid-file. Both are small extensions of the same loop.

Code: https://github.com/rafay99-epic/interview-transcription-pipeline — `transcriber.py`
(`next_end`, `clamp_segments`, `merge_segments`, `transcribe`) and `api.py` (`/jobs`)
