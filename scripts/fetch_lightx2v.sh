#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-$HOME/LightX2V-V100}"
COMMIT="23e83d915fea6c1f237af92ad407abde8f9fba73"
URL="https://github.com/Leonccaa/LightX2V-V100.git"

if [[ ! -d "$DEST/.git" ]]; then
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin "$COMMIT"
git -C "$DEST" checkout --detach "$COMMIT"
echo "Pinned LightX2V-V100 at $(git -C "$DEST" rev-parse HEAD)"
echo "Use config/lightx2v-v100-tp4-16gb.experimental.json as an unvalidated starting point."
