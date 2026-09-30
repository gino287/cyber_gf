#!/usr/bin/env bash
# Download LiveTalking's wav2lip256 weights + official sample avatar (Google Drive folder linked in the
# LiveTalking README) and put them where LiveTalking expects them. Run after lt_install.sh.
set -e
source "$(dirname "$0")/../../config/paths.env"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$LT_WEIGHTS_ORIG")" && cd "$(dirname "$LT_WEIGHTS_ORIG")"
[ -f "$LT_WEIGHTS_ORIG/wav2lip256.pth" ] || uvx gdown --folder "https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ"
cd "$LT_DIR"
cp "$LT_WEIGHTS_ORIG/wav2lip256.pth" models/wav2lip.pth            # LiveTalking hard-codes models/wav2lip.pth
tar -xzf "$LT_WEIGHTS_ORIG/wav2lip256_avatar1.tar.gz" -C data/avatars/
ls -lh models/wav2lip.pth; ls data/avatars/
