# Step 12 MVP: voice pipeline → LiveTalking (no streaming)

```
assets/test.wav → Faster-Whisper large-v3 → llama.cpp Qwen3-14B (:8090) → faster-qwen3-tts (cloned voice)
   → complete reply.wav → POST /humanaudio → LiveTalking Wav2Lip (wav2lip256_myavatar) → existing WebRTC page
```

## How to run
1. llama-server running on :8090 (`scripts\windows\start_llama.bat`).
2. LiveTalking running in **Ubuntu-24.04** with `--avatar_id wav2lip256_myavatar` (port 8010).
3. Browser: http://localhost:8010/index.html → connect. The avatar video must be playing. Keep the page open.
4. Run one command:
   - Windows: `scripts\windows\run_step12_mvp.bat`
   - Inside Ubuntu-24.04: `bash <repo-in-wsl>/scripts/wsl/run_step12_mvp.sh`
   - Options: `--in other.wav`, `--session <id>` (overrides auto-discovery; also `LT_SESSION_ID`).

Both launchers force **Ubuntu-24.04**: the `.bat` uses `wsl -d Ubuntu-24.04`, and the `.sh` and `.py` abort when `WSL_DISTRO_NAME` is anything else.

Files: `scripts/wsl/step12_mvp.py`, `scripts/wsl/run_step12_mvp.sh`, `scripts/windows/run_step12_mvp.bat`, `config/step12.json`.
Outputs per run: `outputs/step12/<YYYYmmdd_HHMMSS>/` containing `transcript.txt`, `reply.txt`, `reply.wav`, `timings.json`.

## LiveTalking API used (installed commit b3e7490; source not modified)
| Call | Where in LiveTalking | Use |
|---|---|---|
| `GET /api/admin/sessions` | `server/routes.py:admin_sessions` | Lists active sessions (`sessionid`, `avatar_id`, `speaking`). Used to **auto-discover** the browser's WebRTC session. |
| `POST /humanaudio` (multipart/form-data: text field `sessionid`, file field `file`) | `server/routes.py:humanaudio` → `BaseAvatar.put_audio_file` | Sends the complete reply WAV. Returns `{"code":0,"msg":"ok"}`, or `{"code":-1,"msg":"session not found"}` with HTTP 200. |
| `POST /is_speaking` (JSON `{"sessionid"}`) | `server/routes.py:is_speaking` | Polled every 40 ms to timestamp speaking start and end. |

**Audio format:** LiveTalking reads the upload with `soundfile.read` (any format soundfile supports). It keeps only the first channel and **resamples to 16 kHz with `resampy`** when needed. We send our 24 kHz mono 16-bit `reply.wav` unchanged; its log shows `put audio stream 24000: (374400,)` → `resampling into 16000`. The audio is split into 20 ms chunks (320 samples) and fed to the Wav2Lip ASR/feature queue, and playback is paced by the WebRTC track.

**Session handling:**
- A session is created when the browser page connects (`POST /offer`, uuid4).
- It is removed automatically when the WebRTC connection closes or fails.
- The script takes all sessions whose `avatar_id` equals `config/step12.json → livetalking.avatar_id`:
  - exactly one: use it;
  - several: use the newest (dict insertion order) and print the list;
  - none: stop and print instructions.
- Nothing is hard-coded, and `--session` overrides the choice.

## First end-to-end run (2026-09-28 23:42, `outputs/step12/20260928_234255/`)
- Transcript: 你今天过得怎么样?有没有什么有趣的事情?
- Reply: 今天还行呀，就是工作有点忙。对了，中午吃饭的时候看到一只小猫咪在路边转悠，我差点就把它抱回家了，不过想了想还是算了，怕你又说我养猫不养你了。
- `reply.wav`: 15.6 s, 24 kHz, cloned voice (fast backend, RTF 0.305)

| Timestamp (s from input start) | Value | Stage duration |
|---|---|---|
| ASR complete | 1.066 | ASR 1.07 s (3.84 s input) |
| LLM first token | 1.193 | 0.13 s after ASR |
| LLM complete | 1.804 | LLM 0.74 s |
| TTS complete (full reply.wav) | 6.563 | TTS 4.76 s for 15.6 s of audio |
| /humanaudio POST complete | 6.728 | 0.17 s |
| **LiveTalking speaking start** | **6.909** | 0.18 s after POST |
| LiveTalking speaking end | 22.446 | spoke 15.54 s ≈ audio length 15.6 s |

One-time setup per command (not in the table): Whisper load 4.1 s, TTS load + CUDA-graph capture 12.5 s, TTS warm-up 1.4 s. The warm-up synthesizes 「嗯。」 to cache the reference prompt, and the output is discarded.

LiveTalking log for this run: `put audio stream 24000: (374400,)` → `resampling into 16000` → `状态切换：静音 → 说话` → `说话 → 静音`.
**Visual lip sync must be confirmed by the user in the browser.**

## Is everything local?
- **ASR, LLM, TTS and lip sync all run on this machine.** The WAV goes WSL → `localhost:8010`, and llama.cpp is reached at `localhost:8090` (WSL mirrored networking).
- **Edge TTS is not used.** LiveTalking is still configured with `tts=edgetts`, but that only serves its own `/human` text route, which this pipeline never calls.
- **Not strictly offline:**
  - WebRTC ICE uses the STUN server `stun.l.google.com:19302` (the article's setting) to gather candidates. It carries no audio; media flows browser ↔ localhost.
  - The HF libraries may make metadata requests to huggingface.co when loading models. They send no audio, and `HF_HUB_OFFLINE=1` would stop them.

## What remains before streaming (chunk 8)
1. **TTS:** switch to `generate_voice_clone_streaming(chunk_size=8)` (benchmarked: first chunk ≈ 0.31 s, no stalls).
2. **Transport:** `/humanaudio` takes a whole file, and each call emits its own `start` / `end` events (it is not designed for continuous chunks). Options:
   - POST each 0.64 s chunk as its own WAV (simplest; may cause mouth-closing or state flicker at boundaries);
   - LiveTalking's frame-level path (`put_audio_frame`, 16 kHz 20 ms float32), which has no HTTP route today, so it would need an adapter or a patch — **to be discussed before touching LiveTalking**;
   - a WebSocket or custom endpoint.
3. **Ordering and back-pressure:** send chunks in order without flooding LiveTalking's queue; handle the last chunk's `end` event.
4. **LLM → TTS overlap:** start TTS per sentence while the LLM is still generating (today we wait for the full reply).
5. **Measure:** time from input end to avatar speaking start (MVP: 6.9 s, dominated by full-reply TTS 4.8 s) and check for mouth or audio gaps between chunks.
6. Later (not in scope): microphone/VAD, interruption (`/interrupt_talk` exists), persistent model server (setup currently takes about 18 s per command).
