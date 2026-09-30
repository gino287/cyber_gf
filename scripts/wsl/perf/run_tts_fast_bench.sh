#!/usr/bin/env bash
# Benchmark faster-qwen3-tts (optimized backend) in its own env. Needs llama-server running for VRAM parity.
source "$(dirname "$0")/../../../config/paths.env"
source "$TTS_FAST_VENV/bin/activate"
export HF_HOME
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
python "$PROJECT_WSL/scripts/wsl/perf/tts_fast_bench.py" "$@"
