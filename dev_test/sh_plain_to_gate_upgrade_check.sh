#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_plain_to_gate_upgrade"
mkdir -p "$tmp_dir"

plain_init="$tmp_dir/plain_init.mtp"
train="$tmp_dir/one.cfg"
upgraded="$tmp_dir/upgraded_shared_gate.mtp"
upgrade_log="$tmp_dir/upgrade_train.log"
rm -f "$plain_init" "$train" "$upgraded" "$upgrade_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$plain_init" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0

if grep -q "sh_scalar_info" "$plain_init"; then
  echo "plain init-sh unexpectedly wrote sh_scalar_info metadata" >&2
  exit 1
fi

./bin/mlp-sus2 train "$plain_init" "$train" \
  --trained-pot-name="$upgraded" \
  --max-iter=1 \
  --skip-preinit \
  --do-lin \
  --do-lin-rescale \
  --fine-tune \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$upgrade_log"

grep -q "SUS2-SH plain-to-gate upgrade enabled" "$upgrade_log"
grep -q "plain-to-gate upgrade disabled do-lin/do-lin-rescale/fine-tune" "$upgrade_log"
if grep -q "fine-tune mode enabled" "$upgrade_log"; then
  echo "plain-to-gate upgrade should disable fine-tune rescale/linear warmup" >&2
  exit 1
fi
if grep -q "Fine-tune initial" "$upgrade_log"; then
  echo "plain-to-gate upgrade ran fine-tune initial rescale/linear warmup" >&2
  exit 1
fi

grep -q "sh_scalar_info_count" "$upgraded"
grep -q "two_layer_gate_enabled = true" "$upgraded"
grep -q "two_layer_gate_body_order_max = 3" "$upgraded"
grep -q "two_layer_gate_include_one_body = false" "$upgraded"
grep -q "two_layer_gate_radial_mode = shared-radial" "$upgraded"
grep -q "two_layer_gate_weights" "$upgraded"
