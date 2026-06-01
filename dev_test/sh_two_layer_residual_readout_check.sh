#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_residual_readout"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/residual_init.mtp"
nonzero_model="$tmp_dir/residual_nonzero.mtp"
train="$tmp_dir/one.cfg"
rm -f "$init_model" "$nonzero_model" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-shared-radial \
  --two-layer-residual

python3 - "$init_model" "$nonzero_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()

def replace_list(name, updater):
    global text
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    updater(values)
    replacement = name + " = {" + ", ".join(f"{x:.15e}" for x in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

def replace_or_append_list(name, values):
    global text
    replacement = name + " = {" + ", ".join(f"{x:.15e}" for x in values) + "}"
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text)
    if match:
        text = text[:match.start()] + replacement + text[match.end():]
    else:
        text = text.rstrip() + "\n" + replacement + "\n"

def set_prefix(values, prefix):
    if len(values) < len(prefix):
        raise SystemExit("coefficient list is shorter than expected")
    for i, value in enumerate(prefix):
        values[i] = value

e0_match = re.search(r"two_layer_residual_e0_coeffs = \{([^}]*)\}", text)
if not e0_match:
    raise SystemExit("missing two_layer_residual_e0_coeffs")
e0_count = len([x for x in e0_match.group(1).split(",") if x.strip()])
if e0_count < 3:
    raise SystemExit("expected at least three E0 coefficients")

replace_or_append_list("species_coeffs", [1.25, 0.80])
moments = [0.0] * e0_count
set_prefix(moments, [0.20, -0.10, 0.05])
replace_or_append_list("moment_coeffs", moments)
replace_list("two_layer_gate_weights", lambda v: set_prefix(v, [0.35, -0.20, 0.15]))
replace_list("two_layer_residual_e0_coeffs", lambda v: set_prefix(v, [0.25, -0.10, 0.05]))
open(target, "w").write(text)
PY

./bin/mlp-sus2 check-linear-readout-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --max-atoms=2 \
  --abs-tolerance=2.0e-8 \
  --rel-tolerance=2.0e-8

./bin/mlp-sus2 check-linear-components-fd-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --max-atoms=1 \
  --displacement=1.0e-5 \
  --abs-tolerance=5.0e-6 \
  --rel-tolerance=5.0e-4
