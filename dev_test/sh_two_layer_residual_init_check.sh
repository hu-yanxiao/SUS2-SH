#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_residual_init"
mkdir -p "$tmp_dir"

model="$tmp_dir/residual.mtp"
saved="$tmp_dir/residual.saved.mtp"
train="$tmp_dir/one.cfg"
rm -f "$model" "$saved" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$model" \
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

grep -q "two_layer_gate_enabled = true" "$model"
grep -q "two_layer_residual_enabled = true" "$model"
grep -q "two_layer_gate_scale_mode = direct" "$model"
grep -q "two_layer_gate_bias = 1" "$model"
grep -q "two_layer_residual_e0_coeff_count" "$model"
grep -q "two_layer_residual_e0_coeffs" "$model"

./bin/mlp-sus2 train "$model" "$train" \
  --max-iter=1 \
  --skip-preinit \
  --trained-pot-name="$saved" \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >/dev/null

grep -q "two_layer_residual_enabled = true" "$saved"
grep -q "two_layer_gate_scale_mode = direct" "$saved"
grep -q "two_layer_residual_e0_coeffs" "$saved"
