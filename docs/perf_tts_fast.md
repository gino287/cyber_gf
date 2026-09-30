# Optimized Qwen3-TTS backend evaluation: faster-qwen3-tts (2026-09-28)

## Chosen implementation
**`andimarafioti/faster-qwen3-tts` 0.5.2** (https://github.com/andimarafioti/faster-qwen3-tts, MIT, 1.4k★, active; PyPI `faster-qwen3-tts`)

Why this one:
- It is the most-used community project that targets exactly our measured bottleneck (per-step kernel-launch / Python overhead). It publishes RTX 4090 + 1.7B results: 4.22× real time, TTFA 174 ms.
- It supports **Base voice cloning with `ref_audio` + `ref_text` (ICL)** as its default mode, which is what we use.
- It keeps the official model code. Its dependency `qwen-tts-hf` 0.1.1.post1 is the official `qwen_tts` package plus the Transformers-5 patch from upstream PR QwenLM/Qwen3-TTS#360.
- It includes an official streaming API and parity tests against upstream.

Alternatives not chosen:
- `tsdocode/nano-qwen3tts-vllm` (138★): a smaller project and a separate engine.
- vLLM-Omni: officially offline-only, with no streaming yet. Kept as the second option, and not needed.

## What it changes vs official qwen-tts 0.1.1
| Area | Official qwen-tts 0.1.1 | faster-qwen3-tts |
|---|---|---|
| Prefill | HF `talker.generate()` | Same model `forward` (dynamic cache), then KV copied into a StaticCache |
| KV cache | HF DynamicCache | **transformers `StaticCache`** (fixed `max_seq_len=2048`) |
| Decode loop | HF `generate()` for the talker, plus a nested HF `generate()` on the code predictor per frame | **Custom Python loop**: talker step and 15-step predictor each **captured as one `torch.cuda.CUDAGraph`** and replayed |
| Sampling | HF logits processors | Re-implemented in HF order (suppress → temperature → top-k → top-p → multinomial), **same defaults**: temperature 0.9, top_k 50, top_p 1.0, repetition penalty 1.05, min 2 new tokens, same suppressed ids |
| torch.compile | no | **no** |
| flash-attn | optional | **not required** (`sdpa`) |
| Streaming | none | `generate_voice_clone_streaming(chunk_size=N)`: codec decoded in chunks with 25-frame left context |
| Reference audio | as given | appends 0.5 s silence by default; **we set `append_silence=False` = upstream behaviour** |
| Numerics | — | Same algorithm; not bit-identical (different SDPA kernel and reduction order with a static mask) |

## Environment (separate; original kept as fallback)
- New: WSL `/root/cyber_gf/.venv-tts-fast`, installed by `scripts/wsl/tts_fast_install.sh`. Contents: torch 2.8.0+cu128 (same as before), faster-qwen3-tts 0.5.2, qwen-tts-hf 0.1.1.post1, **transformers 5.15.1**, faster-whisper 1.2.1.
- Unchanged: `/root/cyber_gf/.venv-voice` (qwen-tts 0.1.1 + transformers 4.57.3), the original Step 9 backend.
- Models: same HF cache, same `Qwen/Qwen3-TTS-12Hz-1.7B-Base` files (no new download).
- Install issue: the resolver picked transformers 5.17.0, which breaks `qwen-tts-hf` (`'MimiConfig' object has no attribute 'rope_theta'` while building the codec's rotary embedding). **Fix: pinned transformers 5.15.1**, the version upstream validated (issue #135). No code was patched.

## Benchmark: identical conditions
Fixed text 「今天嘛，就是平平淡淡过日子呀。倒是你，有没有想我啊？我可是很想你呢。」, the same `assets/ref.wav` + `config/ref_text.txt`, ICL, seed 1234, bf16, sdpa, 24 kHz output, Faster-Whisper large-v3 loaded in the same process, llama-server running.
Scripts: `scripts/wsl/perf/tts_perf.py` (baseline), `scripts/wsl/perf/tts_fast_bench.py` (optimized). Raw data: `outputs/perf/tts_perf.json`, `outputs/perf_fast/fast_bench.json`.

| Metric | Original qwen-tts 0.1.1 | **faster-qwen3-tts (non-streaming)** | **faster-qwen3-tts streaming, chunk 8** | **streaming, chunk 4** |
|---|---|---|---|---|
| Synthesis time (warm, 3 runs / 2 runs) | 24.9–26.1 s | **2.46–2.54 s** | 2.76–2.79 s | 3.23–3.33 s |
| Output audio | 8.16 s | 8.48 s | 8.48 s | 8.48 s |
| **RTF** (gen / audio) | **3.05–3.19** | **0.290–0.299** | 0.325–0.329 | 0.381–0.393 |
| Speed vs real time | 0.32× | **3.4×** | 3.1× | 2.6× |
| **Time to first audio** | 24.9 s (no streaming) | = full synthesis | **0.307–0.317 s** (0.64 s chunk) | **0.223–0.243 s** (0.32 s chunk) |
| Playback stalls (simulated, starting at the first chunk) | would stall every frame | — | **0** | **0** |
| GPU utilization (avg) | 26–28% | **84–88%** | 77–80% | 71–73% |
| SM clock / power | ~1.5 GHz / ~62 W | 2.73 GHz / ~181 W | 2.73 GHz / ~180 W | 2.73 GHz / ~176 W |
| Process CPU | 100% (1 core) | 100% (1 core) | 100% | 100% |
| One-time setup | load 8.5 s, prompt 1.14 s | load 9.7 s, **CUDA-graph capture 2.5 s**, prompt ~1.1 s | same | same |

**Speed-up: about 10.6× in RTF. First audio arrives about 0.3 s after the request (streaming, chunk 8), and playback never stalls.**

### Voice quality (scored identically for both backends)
Speaker similarity = cosine between Qwen3-TTS speaker-encoder x-vectors of each output and `ref.wav`. Intelligibility = Whisper large-v3 round-trip, character error rate vs the fixed text (punctuation ignored).

| Output | Speaker similarity to ref | CER |
|---|---|---|
| ref.wav (self) | 1.0000 | — |
| Original backend `fixed_run1` / `fixed_run2` | 0.9935 / 0.9935 | 0.0 / 0.0 |
| Original Step 9 `reply.wav` (the one you approved) | 0.9923 | — (different text) |
| **Fast non-stream run1 / run2** | **0.9925 / 0.9925** | **0.0 / 0.0** |
| **Fast stream chunk 8** | **0.9924** | **0.0** |
| **Fast stream chunk 4** | **0.9923** | **0.0** |

The difference in speaker similarity is at most 0.001, within the range between different original-backend outputs, and pronunciation is perfect in every case. **Listen for yourself:** `outputs/perf_fast/fast_nonstream_run1.wav` and `fast_stream_cs8_run1.wav` vs `outputs/perf/fixed_run1.wav`.

### VRAM (whole GPU)
Note: during this run a **LiveTalking `app.py` (wav2lip256_myavatar) was already running**. It was started at 23:16, not by this benchmark, and was left untouched. It uses about 1.6 GB (visible as the WSL `vmwp` counter rising from 1.0 to 2.6 GB).

| Stage | Original run | Fast run (with LiveTalking already up) |
|---|---|---|
| Before Python models (desktop + llama [+ LiveTalking]) | 12,585 MiB | 14,182 MiB |
| + Whisper | +4,000 | +4,000 |
| + TTS weights | +4,196 | +4,196 |
| + CUDA-graph capture / static cache | — | +292 |
| Peak during generation | 21,339 (+558) | **23,300** (+630) |

- The fast backend costs about **+360 MiB more** than the original (static KV cache plus graph memory pools).
- **Full stack measured: llama + Whisper + TTS + LiveTalking + desktop = 23.3 GB of 24 GB.** It fits, with about 1.2 GB of headroom.

## Result vs success criteria
| Criterion | Result |
|---|---|
| Minimum: RTF < 1.0 | ✅ 0.29 |
| Target: clearly faster than real time | ✅ 3.4× |
| Preferred: first audio within a few hundred ms | ✅ 0.31 s (chunk 8), 0.22–0.24 s (chunk 4) |
| Voice quality comparable | ✅ speaker similarity equal within 0.001, CER 0 (confirm by listening) |

Compared with the upstream README for RTX 4090 + 1.7B (4.22× real time, TTFA 174 ms), we get 3.4× and about 310 ms. The gap plausibly comes from WSL2 overhead and LiveTalking running alongside.
