# Qwen3-TTS performance investigation (2026-09-28)

Measurement only. No change to models, voice, prompt, output quality, LiveTalking, llama.cpp or ComfyUI.
Nothing in site-packages was edited: timing used wrappers on the loaded instance plus forward hooks.

- Scripts: `scripts/windows/perf_vram_stages.ps1` (stages 1–2, per-process VRAM, launches the rest), `scripts/wsl/perf/tts_perf.py`
- Raw data: `outputs/perf/tts_perf.json`, `stages_windows.json`, `gpu_processes_during_tts.json`, `tts_perf.log`
- Audio to listen to: `outputs/perf/fixed_run{1,2,3}.wav` (identical, same seed)

Fixed text for all runs: 「今天嘛，就是平平淡淡过日子呀。倒是你，有没有想我啊？我可是很想你呢。」
Same `assets/ref.wav` + `config/ref_text.txt`, seed 1234, bf16, `sdpa`, default generation config.

---

## A. VRAM breakdown (whole GPU, `nvidia-smi`)

| Stage | Total used | Delta | Notes |
|---|---|---|---|
| 1. Baseline, no cyber_gf model | **2,560 MiB** | — | Windows desktop and apps (see below) |
| 2. + llama.cpp Qwen3-14B Q4_K_M (`-c 8192`) | 12,590 | **+10,030** | Per-process counter: llama-server 10,020 MiB |
| 3. + Faster-Whisper large-v3 (CT2 float16) | 16,585 | **+3,995** | Includes this process's CUDA context and cuBLAS workspace |
| 4. + Qwen3-TTS 1.7B loaded, before generation | 20,781 | **+4,196** | torch allocated 4,007 / reserved 4,134 MiB |
| 5. Peak during TTS generation | **21,339** | **+558** | torch peak allocated 4,341, reserved 4,672 MiB |
| 6. After generation finishes | 21,339 | 0 | Caching allocator keeps the reserved blocks |
| 7. (Informational) Whisper deleted | 17,759 | **−3,580** | What unloading Whisper would free |

Qwen3-TTS weights by component (all on `cuda:0`):

| Component | Parameters | Size |
|---|---|---|
| talker (28 layers) | 1,741.6 M | 3,322 MiB |
| code_predictor (5 layers) | 175.1 M | 334 MiB |
| speech_tokenizer (12 Hz codec) | 170.6 M | 325 MiB |
| speaker_encoder | 12.0 M | 23 MiB |
| **Total** | **1,928.7 M** | **3,679 MiB** |

**Unrelated GPU users (Windows per-process "Dedicated Usage" counters).** These counters overlap and over-count, so they don't add up to the total. They are only useful for telling which processes hold VRAM.
- Two `dwm.exe` instances (4.2 GB and 3.8 GB) plus a `LogonUI` process, meaning a **second Windows session** exists.
- `AnyDesk` (remote desktop).
- `chrome` ×2 and `msedgewebview2` ×3.
- `BaiduNetdiskRender` / `BaiduNetdiskUnite`.
- `WindowsTerminal`, `explorer`, `csrss`.
- `vmwp` 1,016 MiB: the WSL VM. During TTS a second `vmwp` counter shows 8,752 MiB, which is our WSL Python process (Whisper + TTS).
- CosyVoice (:8001) and the :8000 server are stopped and don't appear.

## B. Qwen3-TTS runtime verification

| Item | Result |
|---|---|
| Device placement | Every parameter and buffer of the talker, code_predictor, speech_tokenizer and speaker_encoder is on `cuda:0` |
| dtype | bf16 for all weights (fp32 only in a few tiny buffers). **bf16 is active.** |
| Attention implementation | `sdpa` for the top model, talker, code_predictor and codec decoder. The `flash-attn` package is not installed. |
| SDPA kernel actually used | Profiler: **21,068 calls to PyTorch's built-in FlashAttention** (`pytorch_flash::flash_fwd_kernel`), 16 memory-efficient calls, 0 math |
| Attention cost | 84 ms of 6,278 ms total GPU kernel time (**1.3%**) |
| CPU fallback | **None found.** CPU-only aten ops in the profile are metadata ops only (`view`, `as_strided`, `transpose`, `empty_strided`, `_to_copy`). All compute kernels run on the GPU. |
| GPU utilization during synthesis | nvidia-smi average **26–28%** (max 62–71%), SM clock ~1.4–1.7 GHz, **~60–67 W** of 450 W |
| GPU busy fraction (profiler) | Kernel time 6.28 s over 36.6 s wall = **17%** |
| CPU utilization during synthesis | Process **100.0%, which is exactly one core saturated**. The system average is 11% of 12 cores. |
| Kernel launches per synthesis | **709,365** `cudaLaunchKernel` calls, about **7,000 per audio frame** |
| Host–device sync points | 3,581 `.item()` and 3,696 `cudaStreamSynchronize` (about 35 per frame) |
| Hottest kernels | bf16 **gemv** (batch-1 matrix-vector) 3.8 s, then many tiny elementwise / RMSNorm / copy / cat kernels |
| Generation parameters | Checkpoint `generate_config`: `do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05`, the same for the sub-talker, `max_new_tokens=8192` |
| Structure (speed-relevant) | 12.5 frames per second of audio × 16 codebooks. Per frame: 1 talker forward (28 layers) + **one HuggingFace `generate()` call on the code_predictor with 15 sequential forwards** (measured: 103 talker forwards and 1,530 code_predictor forwards for 102 frames) |
| TF32 | Off (irrelevant for bf16) |
| `flash-attn is not installed` warning | Printed at import by `qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py`, part of the **25 Hz tokenizer that this 12 Hz model doesn't use** |

## C. Fixed benchmark

One-time costs: model load 8.5 s (Whisper 4.0 s, from disk cache), voice-clone prompt 1.14 s (76 reference frames).

| Run | TTS time | Audio | RTF | codec generate | codec decode | ms / frame (real time = 80) | GPU util avg | Process CPU | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 24.93 s | 8.16 s | 3.06 | 24.81 s | 0.12 s | 243 | 25.8% | 100.6% | 21,337 MiB |
| run 1 | 25.35 s | 8.16 s | **3.11** | 25.25 s | 0.10 s | 248 | 26.2% | 100.0% | 21,337 |
| run 2 | 24.89 s | 8.16 s | **3.05** | 24.79 s | 0.10 s | 243 | 26.4% | 100.0% | 21,339 |
| run 3 | 26.06 s | 8.16 s | **3.19** | 25.98 s | 0.07 s | 255 | 28.0% | 100.2% | 21,339 |

About **99.6% of TTS time is the autoregressive code generation.** Decoding codes to a waveform takes about 0.1 s.

## D. Streaming

**The installed `qwen-tts` 0.1.1 has no official streaming output**, and neither does upstream `main` (also 0.1.1):
- The docstring says `non_streaming_mode=False` "only simulates streaming text input … rather than enabling true streaming input or streaming generation".
- `generate_voice_clone` decodes only after all codes exist.
- `model.generate()` doesn't pass a `streamer` through.
- The official README says vLLM-Omni supports "only offline inference", with streaming coming "later".

So **no streamed WAV can be produced with the official API.** The non-streaming WAVs are in `outputs/perf/`.

These are lower bounds measured from the actual generation timeline (last run). They're estimates, not official streaming results:

| Metric | Value |
|---|---|
| Prefill (prompt processed) | 0.07 s |
| First codec frame (80 ms of audio) | 0.32 s |
| 6 frames ready (0.48 s audio) | 1.54 s; decoding them takes 0.04 s, so **estimated first audio ≈ 1.58 s** |
| 12 frames ready (0.96 s audio) | 2.96 s; decode 0.04 s, so estimated ≈ 3.0 s |
| Could playback run continuously? | **No.** Each 80 ms frame takes about 245 ms to produce, so a player would stall about every 80 ms of audio unless it pre-buffers about 68% of the reply first. |

## E. Comparison with the article

**VRAM (article about 17–18 GB for the whole setup vs ours 21.3–21.8 GB peak).** Comparing like for like:

| | Article | Ours (measured) | Difference |
|---|---|---|---|
| Desktop / other apps baseline | not counted | 2.56 GB | **+2.56** |
| LLM (Qwen3-14B Q4_K_M) | ~9 GB | 10.03 GB (weights ~8.4 + f16 KV cache for 8192 ctx ~1.3 + compute buffers) | +1.0 |
| Whisper large-v3 | ~3 GB | 4.0 GB (weights ~3.1 + CUDA context and cuBLAS workspace) | +1.0 |
| Qwen3-TTS 1.7B | ~4 GB | 4.2 GB loaded + 0.56 GB during generation | +0.8 |
| LiveTalking wav2lip | ~1.3 GB | not loaded in this test | −1.3 |

- **The biggest single difference is the 2.56 GB baseline** (second Windows session, AnyDesk, Chrome, Baidu Netdisk and so on), which the article's table doesn't include.
- The rest is about +0.8–1.0 GB per model: the article's numbers look like rounded weight sizes, and ours include CUDA context, KV cache and workspace.
- Excluding the baseline, our three models peak at **18.8 GB**. That is consistent with the article's 17 GB (which includes LiveTalking) plus the per-component overheads.
- The 21.8 GB seen in the first Step 9 test was a longer reply with a different baseline.

**TTS speed (article "about 300 ms synthesis", near real time, vs our RTF about 3.1).**
1. **Measured cause: we are limited by the CPU launching GPU kernels, not by GPU compute.**
   - The GPU does real work only 17% of the time, at about 60 W, while one CPU core is pinned at 100%.
   - Each frame needs about 7,000 kernel launches and about 35 host–device syncs. Most of that is the per-frame HuggingFace `generate()` on the code_predictor: 15 sequential tiny forwards, each with sampling, logits processors and a stop check that forces a sync.
   - About 245 ms per frame ≈ 7,000 launches × ~35 µs. Eager PyTorch plus Python dispatch in WSL2 (GPU paravirtualisation adds launch latency) fits this cost profile.
2. **Not the cause:**
   - Attention backend: already FlashAttention, only 1.3% of GPU time.
   - Precision: bf16 is active.
   - CPU fallback: none.
   - VRAM pressure: 21.3 of 24 GB, no system-memory fallback.
   - Codec decode: 0.1 s.
3. **Why the article's number differs (can't be verified).** The author's `install-voice.sh` "applies patches" that we never saw, and the article's 300 ms is a rough per-stage estimate, not a measured RTF. The article also says they wait for the whole audio before lip sync. **Nothing measured here shows that the stock `qwen-tts` 0.1.1 eager runtime can reach near-real-time on this machine.**

## Recommended next optimizations (priority order, none applied yet)

1. **Cut the per-step CPU overhead of the talker / code_predictor loop.** This is the only lever aimed at the measured bottleneck. The GPU is 83% idle, so the theoretical headroom is several times the current speed. Candidates:
   - (a) **Official vLLM-Omni** Qwen3-TTS path: optimized serving with CUDA graphs. Documented as offline-only for now, so first check that it supports Base voice cloning with our ref prompt.
   - (b) CUDA graphs / `torch.compile(mode="reduce-overhead")` with a static KV cache for the talker and code_predictor. This would need code outside the package, or a patch. **I'll stop and explain before doing any patch.**
2. **Measure the WSL2 penalty:** run the same `tts_perf.py` benchmark with a native Windows Python and the same model and settings. It's cheap, needs no quality change, and shows how much of the launch latency is WSL-specific.
3. **Streaming / sentence chunking:** only worthwhile once RTF < 1. Until then, a player would stall.
   - Sentence-level chunking of the LLM reply (the official API synthesizes each sentence separately, so no patch is needed) would bring first audio forward to the first sentence. At RTF 3.1 that's still several seconds.
4. **Free VRAM if LiveTalking needs room:** close the second Windows session, AnyDesk, Chrome and Baidu Netdisk (up to about 1–2.5 GB). Unloading Whisper frees 3.6 GB but costs about 4 s to reload per turn, so it isn't recommended for real-time use.
5. **`flash-attn`: lowest priority.** SDPA already runs FlashAttention kernels and attention is 1.3% of GPU time. The warning comes from an unused module.
