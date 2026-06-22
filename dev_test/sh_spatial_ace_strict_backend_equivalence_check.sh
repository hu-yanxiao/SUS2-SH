#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-strict-spatial-ace-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

compare_numeric_cfgs() {
  local base=$1
  local strict=$2
  python3 - "$base" "$strict" <<'PY'
import re
import sys

float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [float(x) for x in float_re.findall(fh.read())]

base = values(sys.argv[1])
strict = values(sys.argv[2])
if len(base) != len(strict):
    raise SystemExit(f"numeric token count mismatch: {len(base)} vs {len(strict)}")

max_abs = 0.0
for i, (a, b) in enumerate(zip(base, strict)):
    diff = abs(a - b)
    max_abs = max(max_abs, diff)
    if diff > 1.0e-10:
        raise SystemExit(f"token {i} differs: {a} vs {b}, diff={diff}")
print(f"strict spatial ACE equivalence passed, max_abs={max_abs:.3e}")
PY
}

run_case() {
  local tag=$1
  local lmax=$2
  local kmax=$3
  local mode=$4
  local body_order=$((kmax + 1))
  local model="$WORKDIR/${tag}.mtp"
  local base_out="$WORKDIR/${tag}.base.cfg"
  local strict_out="$WORKDIR/${tag}.strict.cfg"
  local strict_log="$WORKDIR/${tag}.strict.log"
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

  if [[ "$mode" == "gate" ]]; then
    python3 - "$model" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8").read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if not weights:
    raise SystemExit("empty two_layer_gate_weights")
weights[0] = 2.5e-1
replacement = (
    "two_layer_gate_weights = {"
    + ", ".join(f"{x:.15e}" for x in weights)
    + "}"
)
open(path, "w", encoding="utf-8").write(
    text[:match.start()] + replacement + text[match.end():]
)
PY
  fi

  "$BIN" calc-efs "$model" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$base_out" >/dev/null
  SUS2_SH_STRICT_SPATIAL_ACE=1 SUS2_SH_STRICT_SPATIAL_ACE_TRACE=1 \
    "$BIN" calc-efs "$model" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$strict_out" >"$strict_log"

  grep -q "SUS2-SH strict spatial ACE backend enabled" "$strict_log"
  grep -q "implementation=spatial-grouped-exact" "$strict_log"
  if grep -q "implementation=product-row-scaffold" "$strict_log"; then
    echo "strict spatial ACE backend is still using the product-row scaffold" >&2
    exit 1
  fi
  if [[ "$mode" == "gate" ]]; then
    grep -q "SUS2-SH strict spatial ACE gate scalar backend enabled" "$strict_log"
    grep -q "implementation=spatial-grouped-exact-subset" "$strict_log"
  fi
  compare_numeric_cfgs "$base_out" "$strict_out"
}

run_case l2k2_plain 2 2 plain
run_case l3k3_plain 3 3 plain
run_case l4k4_plain 4 4 plain
run_case l2k2_gate 2 2 gate
run_case l3k3_gate 3 3 gate
run_case l4k4_gate 4 4 gate
