#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_forward"
mkdir -p "$tmp_dir"

zero_model="$tmp_dir/gated_zero.mtp"
nonzero_model="$tmp_dir/gated_nonzero.mtp"
zero_pred="$tmp_dir/zero_pred.cfg"
nonzero_pred="$tmp_dir/nonzero_pred.cfg"
train="$tmp_dir/one.cfg"
rm -f "$zero_model" "$nonzero_model" "$zero_pred" "$nonzero_pred" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$zero_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-body-order=3

python3 - "$zero_model" "$nonzero_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if not weights:
    raise SystemExit("empty two_layer_gate_weights")
weights[0] = 0.25
replacement = "two_layer_gate_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
text = text[:match.start()] + replacement + text[match.end():]
open(target, "w").write(text)
PY

./bin/mlp-sus2 calc-efs "$zero_model" "$train" "$zero_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$nonzero_model" "$train" "$nonzero_pred" >/dev/null

python3 - "$zero_pred" "$nonzero_pred" <<'PY'
import sys

def energy(path):
    expect = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if expect:
                return float(line.split()[0])
            if line == "Energy":
                expect = True
    raise SystemExit(f"missing Energy in {path}")

e0 = energy(sys.argv[1])
e1 = energy(sys.argv[2])
delta = abs(e1 - e0)
if delta <= 1.0e-12:
    raise SystemExit(f"nonzero gate did not change energy: delta={delta}")
PY
