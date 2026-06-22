#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
if [[ ! -x "$BIN" ]]; then
  echo "binary is not executable: $BIN" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-spatial-ace-fd-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

CFG=${SUS2_SH_SPATIAL_ACE_FD_CFG:-"$REPO_ROOT/dev_test/NaCl_small.cfgs"}
COEFF_WINDOW=${SUS2_SH_SPATIAL_ACE_FD_COEFF_WINDOW:-8}
CASE_FILTER=${SUS2_SH_SPATIAL_ACE_CASES:-all}

CASES=(
  "l2k2_plain:2:2:plain"
  "l3k3_plain:3:3:plain"
  "l4k4_plain:4:4:plain"
  "l2k2_gate:2:2:gate"
  "l3k3_gate:3:3:gate"
  "l4k4_gate:4:4:gate"
)

case_enabled() {
  local case_name=$1
  if [[ "$CASE_FILTER" == "all" ]]; then
    return 0
  fi
  [[ ",$CASE_FILTER," == *",$case_name,"* ]]
}

set_gate_weights_nonzero() {
  local model=$1
  python3 - "$model" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8").read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if len(weights) < 3:
    raise SystemExit("expected at least three gate weights")
weights[0] = 3.5e-1
weights[1] = -2.0e-1
weights[2] = 1.5e-1
replacement = (
    "two_layer_gate_weights = {"
    + ", ".join(f"{x:.15e}" for x in weights)
    + "}"
)
open(path, "w", encoding="utf-8").write(
    text[:match.start()] + replacement + text[match.end():]
)
PY
}

run_case() {
  local case_name=$1
  local lmax=$2
  local kmax=$3
  local mode=$4
  local body_order=$((kmax + 1))
  local model="$WORKDIR/${case_name}.mtp"
  local efs_log="$WORKDIR/${case_name}.efs_fd.log"
  local grad_log="$WORKDIR/${case_name}.loss_gradient.log"

  if [[ "$mode" == "gate" ]]; then
    "$BIN" init-sh "$model" \
      --species-count=2 \
      --l-max="$lmax" \
      --k-max="$kmax" \
      --body-order="$body_order" \
      --sh-factor-pruning=q-total \
      --cutoff=7.5 \
      --radial-basis-size=4 \
      --radial-basis-type=RBChebyshev_sss \
      --init-params=same \
      --write-sh-scalar-info \
      --two-layer-gate >/dev/null
    set_gate_weights_nonzero "$model"
  else
    "$BIN" init-sh "$model" \
      --species-count=2 \
      --l-max="$lmax" \
      --k-max="$kmax" \
      --body-order="$body_order" \
      --sh-factor-pruning=q-total \
      --cutoff=7.5 \
      --radial-basis-size=4 \
      --radial-basis-type=RBChebyshev_sss \
      --init-params=same >/dev/null
  fi

  SUS2_SH_STRICT_SPATIAL_ACE=1 SUS2_SH_STRICT_SPATIAL_ACE_TRACE=1 \
    "$BIN" check-efs-fd-dev "$model" "$CFG" \
    --max-configs=1 \
    --max-atoms=2 \
    --displacement=1.0e-5 \
    --abs-tolerance=1.0e-4 \
    --rel-tolerance=1.0e-3 >"$efs_log"

  grep -q "SUS2-SH strict spatial ACE backend enabled" "$efs_log"
  grep -q "implementation=spatial-grouped-exact" "$efs_log"
  if grep -q "implementation=product-row-scaffold" "$efs_log"; then
    echo "$case_name: strict backend is still using the product-row scaffold" >&2
    exit 1
  fi
  grep -q "force_components=6" "$efs_log"
  grep -q "stress_components=9" "$efs_log"

  SUS2_SH_STRICT_SPATIAL_ACE=1 SUS2_SH_STRICT_SPATIAL_ACE_TRACE=1 \
    "$BIN" check-loss-gradient-dev "$model" "$CFG" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=1 \
    --stress-weight=1 \
    --radial-smooth=0 \
    --radial-smooth-grid=128 \
    --scalar-head-l2=0 \
    --gate-scalar-l2=0 \
    --gate-mix-l2=0 \
    --gate-full-l2=0 \
    --displacement=1.0e-6 \
    --abs-tolerance=1.0e-4 \
    --rel-tolerance=5.0e-3 \
    --coeff-start=0 \
    --coeff-end="$COEFF_WINDOW" >"$grad_log"

  grep -q "SUS2-SH strict spatial ACE backend enabled" "$grad_log"
  grep -q "implementation=spatial-grouped-exact" "$grad_log"
  grep -q "checked_coeffs=" "$grad_log"
  echo "$case_name strict spatial ACE FD matrix passed"
}

for entry in "${CASES[@]}"; do
  IFS=: read -r case_name lmax kmax mode <<< "$entry"
  if case_enabled "$case_name"; then
    run_case "$case_name" "$lmax" "$kmax" "$mode"
  fi
done
