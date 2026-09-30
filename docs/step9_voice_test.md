# Step 9 (rebuilt): voice pipeline test, 2026-09-28

Pipeline: `assets/test.wav` → Faster-Whisper large-v3 → llama.cpp Qwen3-14B (:8090) → Qwen3-TTS 1.7B Base voice clone (`assets/ref.wav` + `config/ref_text.txt`) → `outputs/voice_test/reply.wav`.
No microphone, WebRTC, LiveTalking or avatar-sync (as instructed).

## What was installed
| Item | Where |
|---|---|
| Python 3.12 venv | WSL `/root/cyber_gf/.venv-voice` |
| torch / torchaudio 2.8.0+cu128 | venv |
| qwen-tts 0.1.1 (official Qwen), with transformers 4.57.3 and accelerate 1.12.0 | venv |
| faster-whisper 1.2.1 (official SYSTRAN), with ctranslate2 4.8.2 | venv |
| `Systran/faster-whisper-large-v3` (2.9 GB) | WSL `/root/.cache/huggingface/hub` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (4.3 GB) | WSL `/root/.cache/huggingface/hub` |
| flash-attn | **not installed** (as planned; `sdpa` used instead) |

Install script: `scripts/wsl/voice_install.sh`, log `logs/voice_install.log`. No third-party repo was modified.

## Run
```bash
# needs llama-server running (scripts\windows\start_llama.bat)
bash <repo-in-wsl>/scripts/wsl/run_voice_test.sh --runs 2 [--in file.wav]
```
Full results: `outputs/voice_test/timings.json`; log: `logs/voice_test_run1.log`.

## Results (input `assets/test.wav`, 3.84 s)
One-time setup: ASR load 8.2 s, TTS load 25.4 s, voice-clone prompt 1.3 s.

| | Run 1 (cold) | Run 2 (warm) |
|---|---|---|
| ASR text | 你今天过得怎么样?有没有什么有趣的事情? | same |
| ASR done | 1.42 s | **0.84 s** |
| LLM first token (from start) | 1.57 s | **0.97 s** (0.13 s after ASR) |
| LLM done (from start) | 1.86 s | 1.46 s |
| Reply | 今天嘛，就是平平淡淡过日子呀。倒是你，有没有想我啊？我可是很想你呢。 | 今天啊，就是平平淡淡过了呗…你呢，今天有什么好玩的事吗？ |
| Reply audio length | 8.8 s | 14.7 s |
| TTS time | 29.1 s | 47.8 s |
| TTS real-time factor | 3.30 | 3.25 |
| **Total end-to-end** | **30.9 s** | **49.3 s** |

Round-trip check: re-transcribing both reply WAVs with Whisper reproduces the reply text word for word, so the speech is intelligible and complete. Voice similarity to `ref.wav` must be judged by ear.

## VRAM (whole GPU; CosyVoice and the :8000 server stopped)
| State | Used |
|---|---|
| Baseline (desktop) | ~1.5 GB |
| + llama-server | 11.5–12.5 GB |
| + faster-whisper large-v3 loaded | 16.5 GB (≈ +4.0 GB) |
| + Qwen3-TTS 1.7B loaded | 20.7 GB (≈ +4.2 GB) |
| **Peak during runs** | **21.8 GB of 24 GB** (torch reserved 4.9 GB for TTS) |

## Findings
1. **ASR and LLM are fast and match the article.** The article estimates ~300 ms for ASR and ~300 ms for LLM first token. Here, warm, ASR takes 0.84 s for a 3.8 s clip and the LLM's first token arrives 0.13 s after ASR.
2. **TTS is the bottleneck.** Non-streaming Qwen3-TTS runs about 3.3× slower than real time, so each second of reply audio takes about 3.3 s to generate. The article claims ~300 ms synthesis and 1–2 s to first response. That is not reproduced by this configuration.
   - The log shows `Warning: flash-attn is not installed. Will only run the manual PyTorch version.`
   - Generation is non-streaming: the whole reply is synthesized before anything is written. The article also says it waits for the full audio before lip sync.
3. VRAM fits (21.8 GB peak) only because CosyVoice and the :8000 server were stopped. LiveTalking (~1.3 GB per the article) would bring it to about 23 GB, which is tight.

## Options to investigate (not done; need approval)
- Install `flash-attn` (Qwen's recommendation; must compile or find a prebuilt wheel matching torch 2.8 + cu128 + py3.12).
- Use Qwen3-TTS **streaming** generation to get the first audio chunk early, instead of whole-reply synthesis.
- Check GPU utilization during TTS (likely Python-loop / kernel-launch bound rather than compute bound).
- The article's own measure: keep replies to 2–3 sentences (persona already asks for this; run 2's reply was long).
