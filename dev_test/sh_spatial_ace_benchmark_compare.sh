#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 /path/to/main-mlp-sus2 /path/to/spatial-mlp-sus2 cfg outdir [repeats]" >&2
  exit 2
fi

MAIN_BIN=$1
SPATIAL_BIN=$2
CFG=$3
OUTDIR=$4
REPEATS=${5:-1}

if [[ ! -x "$MAIN_BIN" ]]; then
  echo "main binary is not executable: $MAIN_BIN" >&2
  exit 2
fi
if [[ ! -x "$SPATIAL_BIN" ]]; then
  echo "spatial binary is not executable: $SPATIAL_BIN" >&2
  exit 2
fi
if [[ ! -f "$CFG" ]]; then
  echo "cfg file does not exist: $CFG" >&2
  exit 2
fi
if [[ "$REPEATS" -le 0 ]]; then
  echo "repeats should be positive" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

CASES=(
  "l2k2_plain:2:2:plain"
  "l3k3_plain:3:3:plain"
  "l4k4_plain:4:4:plain"
  "l2k2_gate:2:2:gate"
  "l3k3_gate:3:3:gate"
  "l4k4_gate:4:4:gate"
)

CASE_FILTER=${SUS2_SH_SPATIAL_ACE_CASES:-all}
COMPARE_TOL=${SUS2_SH_SPATIAL_ACE_COMPARE_TOL:-1e-10}
TIMINGS="$OUTDIR/timings.csv"
EQUIV="$OUTDIR/equivalence.csv"

printf "case,impl,repeat,seconds,model,cfg,out,log,time_log\n" > "$TIMINGS"
printf "case,repeat,max_abs,numeric_tokens\n" > "$EQUIV"

case_enabled() {
  local case_name=$1
  if [[ "$CASE_FILTER" == "all" ]]; then
    return 0
  fi
  [[ ",$CASE_FILTER," == *",$case_name,"* ]]
}

set_first_gate_weight_nonzero() {
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
}

compare_numeric_cfgs() {
  local case_name=$1
  local repeat=$2
  local base=$3
  local strict=$4
  python3 - "$case_name" "$repeat" "$base" "$strict" "$COMPARE_TOL" "$EQUIV" <<'PY'
import re
import sys

case_name, repeat, base_path, strict_path, tol_text, csv_path = sys.argv[1:]
tol = float(tol_text)
float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [float(x) for x in float_re.findall(fh.read())]

base = values(base_path)
strict = values(strict_path)
if len(base) != len(strict):
    raise SystemExit(f"{case_name}: numeric token count mismatch: {len(base)} vs {len(strict)}")
max_abs = 0.0
for i, (a, b) in enumerate(zip(base, strict)):
    diff = abs(a - b)
    max_abs = max(max_abs, diff)
    if diff > tol:
        raise SystemExit(f"{case_name}: token {i} differs: {a} vs {b}, diff={diff}, tol={tol}")
with open(csv_path, "a", encoding="utf-8") as out:
    out.write(f"{case_name},{repeat},{max_abs:.17g},{len(base)}\n")
print(f"{case_name} repeat {repeat}: max_abs={max_abs:.3e}, numeric_tokens={len(base)}")
PY
}

time_calc_efs() {
  local impl=$1
  local case_name=$2
  local repeat=$3
  local bin=$4
  local model=$5
  local out=$6
  local log=$7
  local time_log=$8

  python3 - "$impl" "$bin" "$model" "$CFG" "$out" "$log" "$time_log" <<'PY'
import os
import subprocess
import sys
import time

impl, bin_path, model, cfg, out, log_path, time_log_path = sys.argv[1:]
env = os.environ.copy()
if impl == "spatial_strict":
    env["SUS2_SH_STRICT_SPATIAL_ACE"] = "1"
    env["SUS2_SH_STRICT_SPATIAL_ACE_TRACE"] = "1"

start = time.perf_counter()
with open(log_path, "wb") as log:
    proc = subprocess.run(
        [bin_path, "calc-efs", model, cfg, out],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
elapsed = time.perf_counter() - start
with open(time_log_path, "w", encoding="utf-8") as handle:
    handle.write(f"real {elapsed:.9f}\n")
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
PY

  local seconds
  seconds=$(awk '/^real / {print $2}' "$time_log" | tail -n 1)
  if [[ -z "$seconds" ]]; then
    echo "could not parse timing from $time_log" >&2
    exit 1
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$case_name" "$impl" "$repeat" "$seconds" "$model" "$CFG" "$out" "$log" "$time_log" \
    >> "$TIMINGS"
}

require_strict_trace() {
  local case_name=$1
  local mode=$2
  local log=$3
  grep -q "SUS2-SH strict spatial ACE backend enabled" "$log"
  grep -q "implementation=spatial-grouped-exact" "$log"
  if grep -q "implementation=product-row-scaffold" "$log"; then
    echo "$case_name: strict backend is still using the product-row scaffold" >&2
    exit 1
  fi
  if [[ "$mode" == "gate" ]]; then
    grep -q "SUS2-SH strict spatial ACE gate scalar backend enabled" "$log"
  fi
  echo "$case_name: strict backend trace verified"
}

for entry in "${CASES[@]}"; do
  IFS=: read -r case_name lmax kmax mode <<< "$entry"
  if ! case_enabled "$case_name"; then
    continue
  fi
  body_order=$((kmax + 1))
  model="$OUTDIR/${case_name}.mtp"

  if [[ "$mode" == "gate" ]]; then
    "$MAIN_BIN" init-sh "$model" \
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
    set_first_gate_weight_nonzero "$model"
  else
    "$MAIN_BIN" init-sh "$model" \
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

  for repeat in $(seq 1 "$REPEATS"); do
    main_out="$OUTDIR/${case_name}.main.r${repeat}.cfg"
    spatial_out="$OUTDIR/${case_name}.spatial_strict.r${repeat}.cfg"
    main_log="$OUTDIR/${case_name}.main.r${repeat}.log"
    spatial_log="$OUTDIR/${case_name}.spatial_strict.r${repeat}.log"
    main_time="$OUTDIR/${case_name}.main.r${repeat}.time"
    spatial_time="$OUTDIR/${case_name}.spatial_strict.r${repeat}.time"

    time_calc_efs main "$case_name" "$repeat" "$MAIN_BIN" "$model" "$main_out" "$main_log" "$main_time"
    time_calc_efs spatial_strict "$case_name" "$repeat" "$SPATIAL_BIN" "$model" "$spatial_out" "$spatial_log" "$spatial_time"
    require_strict_trace "$case_name" "$mode" "$spatial_log"
    compare_numeric_cfgs "$case_name" "$repeat" "$main_out" "$spatial_out"
  done
done

echo "timings: $TIMINGS"
echo "equivalence: $EQUIV"
