#!/usr/bin/env bash
# Headless WebRTC test against a running LiveTalking (start it with lt_test_start.sh first).
# Usage: lt_run_test.sh [tag]   -> sample frames saved to <project>/logs/lt_frames_<tag>/
source "$(dirname "$0")/../../config/paths.env"
TAG="${1:-test}"
OUT="$PROJECT_WSL/logs/lt_frames_$TAG"
[ -d "$LT_DIR" ] && [ -x "$LT_VENV/bin/python" ] || { echo "LiveTalking not found at $LT_DIR -- run inside WSL distro $WSL_DISTRO (wsl -d $WSL_DISTRO)"; exit 1; }
cd "$LT_DIR" && source "$LT_VENV/bin/activate" || exit 1
rm -rf "$OUT"
timeout 60 python "$PROJECT_WSL/scripts/wsl/lt_client_test.py" "http://127.0.0.1:$LT_PORT" "$OUT" 2>&1 | grep -v Warning | tail -n 8
echo "--- server log fps"
grep -i -E "fps" "$LT_LOG_DIR/lt_test.log" | tail -n 6
grep -i -E "error|traceback" "$LT_LOG_DIR/lt_test.log" | tail -n 5
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
