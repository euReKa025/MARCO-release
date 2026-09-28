#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-$ROOT_DIR/../verl}"
VERL_REV=b9d71f9a84ef89ec7f5a946cd277b35165a3daae
if [[ ! -e "$VERL_DIR" ]]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
  git -C "$VERL_DIR" checkout "$VERL_REV"
fi
if [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" != "$VERL_REV" ]]; then
  echo "Expected verl revision $VERL_REV at $VERL_DIR. Set VERL_DIR to a separate checkout." >&2
  exit 1
fi
if git -C "$VERL_DIR" apply --reverse --check "$ROOT_DIR/patches/verl-compat.patch" 2>/dev/null; then
  echo 'MARCO compatibility patch already applied.'
else
  git -C "$VERL_DIR" apply --check "$ROOT_DIR/patches/verl-compat.patch"
  git -C "$VERL_DIR" apply "$ROOT_DIR/patches/verl-compat.patch"
fi
echo "Prepared verl at $VERL_DIR. Install it in the training environment as described in README.md."
