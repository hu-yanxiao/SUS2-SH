#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-spatial-ace-cg-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

LOG="$WORKDIR/cg_map.log"

"$BIN" check-sh-spatial-ace-cg-map-dev \
  --lmax=4 \
  --samples=192 \
  --abs-tolerance=1e-10 \
  --rel-tolerance=1e-9 >"$LOG"

grep -q "strict spatial ACE CG-map check passed" "$LOG"
