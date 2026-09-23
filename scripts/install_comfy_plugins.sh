#!/usr/bin/env bash
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
TARGET="$COMFYUI_DIR/custom_nodes"
mkdir -p "$TARGET"

clone_or_update() {
  local url="$1" name="$2"
  if [[ -d "$TARGET/$name/.git" ]]; then
    git -C "$TARGET/$name" fetch --depth 1 origin
    git -C "$TARGET/$name" reset --hard origin/HEAD
  else
    git clone --depth 1 "$url" "$TARGET/$name"
  fi
  printf '%s\t%s\n' "$name" "$(git -C "$TARGET/$name" rev-parse HEAD)"
}

echo "Installing public V100 components into $TARGET"
clone_or_update https://github.com/rwashy/H3-V100 H3-V100
clone_or_update https://github.com/Amduraznak/minimax-h3-fp16-fix minimax-h3-fp16-fix

echo
echo "Optional GPL-3.0 Ulysses ComfyUI node (full model replicas; not recommended for 4x16GB):"
echo "  git clone --depth 1 https://github.com/dg1kjd/comfyui-v100-sxm2-minimax-h3 $TARGET/comfyui-v100-sxm2-minimax-h3"
echo
echo "Restart ComfyUI before loading a new custom node. Do not stack multiple H3 precision patches on one MODEL branch."

