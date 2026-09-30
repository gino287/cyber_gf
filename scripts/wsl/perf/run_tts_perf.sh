#!/usr/bin/env bash
# Run the Qwen3-TTS performance investigation. Usage: run_tts_perf.sh [--runs N] [--external JSON]
source "$(dirname "$0")/../../../config/paths.env"
source "$VOICE_VENV/bin/activate"
export HF_HOME
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
python "$PROJECT_WSL/scripts/wsl/perf/tts_perf.py" "$@"
