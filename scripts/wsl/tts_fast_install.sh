#!/usr/bin/env bash
# Separate env for the community-optimized Qwen3-TTS backend:
#   faster-qwen3-tts 0.5.2 (https://github.com/andimarafioti/faster-qwen3-tts, MIT)
#   -> qwen-tts-hf 0.1.1.post1 (official qwen_tts + Transformers-5 patch, QwenLM/Qwen3-TTS#360)
# transformers pinned to 5.15.1: 5.17.0 breaks qwen-tts-hf (MimiConfig has no rope_theta); upstream validated 5.15 (issue #135).
# Same torch as .venv-voice (2.8.0+cu128); faster-whisper 1.2.1 added so the pipeline can run in one process.
# Does NOT touch .venv-voice, llama.cpp, ComfyUI or LiveTalking. Models reuse the existing HF cache.
set -ex
source "$(dirname "$0")/../../config/paths.env"
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10
retry(){ for i in $(seq 1 10); do "$@" && return 0; echo "retry $i: $*"; sleep 5; done; return 1; }
[ -d "$TTS_FAST_VENV" ] || uv venv --python 3.12 "$TTS_FAST_VENV"
source "$TTS_FAST_VENV/bin/activate"
retry uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
retry uv pip install faster-qwen3-tts==0.5.2 faster-whisper==1.2.1 torch==2.8.0 torchaudio==2.8.0 transformers==5.15.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
python - <<'PY'
import torch, transformers, faster_qwen3_tts, faster_whisper, ctranslate2, importlib.metadata as md
print("TORCH", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
for p in ("faster-qwen3-tts", "qwen-tts-hf", "transformers", "faster-whisper", "ctranslate2", "huggingface-hub"):
    print(p, md.version(p))
import importlib.util; print("qwen-tts (official, must be absent):", importlib.util.find_spec("qwen_tts") and md.packages_distributions().get("qwen_tts"))
PY
echo TTS_FAST_INSTALL_DONE
