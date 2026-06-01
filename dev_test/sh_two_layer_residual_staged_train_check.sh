#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_residual_staged"
mkdir -p "$tmp_dir"

model="$tmp_dir/staged.mtp"
saved="$tmp_dir/staged.saved.mtp"
log="$tmp_dir/train.log"
train="$tmp_dir/two.cfg"
rm -f "$model" "$saved" "$log" "$train"
awk 'BEGIN{n=0} {print} /^END_CFG/{n++; if (n==2) exit}' example/train.cfg > "$train"

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

./bin/mlp-sus2 train "$model" "$train" \
  --two-layer-residual-staged \
  --stage-a-steps=1 \
  --stage-b-steps=1 \
  --stage-c-steps=1 \
  --skip-preinit \
  --trained-pot-name="$saved" \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 \
  --do-lin \
  --do-lin-steps=1 \
  --do-lin-freq=1 \
  --bfgs-conv-tol=0 >"$log" 2>&1

grep -q "Residual two-layer staged training: stage A" "$log"
grep -q "Residual two-layer staged training: stage B" "$log"
grep -q "Residual two-layer staged training: stage C" "$log"
grep -q "two_layer_residual_enabled = true" "$saved"
grep -q "two_layer_gate_weights" "$saved"
grep -q "two_layer_residual_e0_coeffs" "$saved"
