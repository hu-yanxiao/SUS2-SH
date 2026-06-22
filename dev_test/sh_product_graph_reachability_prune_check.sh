#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./bin/mlp-sus2}
if [[ ! -x "$BIN" ]]; then
  echo "binary is not executable: $BIN" >&2
  exit 2
fi

tmp_dir=".codex_tmp/sh_product_graph_reachability_prune"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

make_model() {
  local out=$1
  shift
  "$BIN" init-sh "$out" \
    --species-count=2 \
    --cutoff=7.5 \
    --radial-basis-size=4 \
    --radial-basis-type=RBChebyshev_sss \
    --init-params=same \
    --write-sh-scalar-info \
    "$@" >/dev/null
}

make_model "$tmp_dir/so3_l3k3_b5.mtp" \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --sh-factor-pruning=q-total

make_model "$tmp_dir/direct_l3k3_b6.mtp" \
  --l-max=3 \
  --k-max=3 \
  --body-order=6 \
  --sh-factor-pruning=q-total \
  --sh-coupling=direct-gaunt

python3 - "$tmp_dir/so3_l3k3_b5.mtp" "$tmp_dir/direct_l3k3_b6.mtp" <<'PY'
import re
import sys

def int_field(text, name):
    match = re.search(rf"^{name}\s*=\s*(\d+)\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def list_field(text, name):
    match = re.search(rf"^{name}\s*=\s*\{{(.*)\}}\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    body = match.group(1).strip()
    if not body:
        return []
    return [int(value) for value in re.findall(r"-?\d+", body)]

def product_list(text):
    match = re.search(r"^sh_products\s*=\s*\{(.*)\}\s*$", text, re.M)
    if not match:
        raise SystemExit("missing sh_products")
    products = []
    for left, right, target, coeff in re.findall(
        r"\{(\d+),\s*(\d+),\s*(\d+),\s*([-+0-9.eE]+)\}",
        match.group(1),
    ):
        products.append((int(left), int(right), int(target), float(coeff)))
    return products

for path in sys.argv[1:]:
    text = open(path, "r", encoding="utf-8").read()
    moment_count = int_field(text, "alpha_moments_count")
    basic_count = int_field(text, "alpha_index_basic_count")
    products = product_list(text)
    scalars = list_field(text, "alpha_moment_mapping")
    required_moments = [False] * moment_count
    for scalar in scalars:
        if not 0 <= scalar < moment_count:
            raise SystemExit(f"{path}: scalar moment out of range: {scalar}")
        required_moments[scalar] = True
    required_products = [False] * len(products)
    for index in range(len(products) - 1, -1, -1):
        left, right, target, _ = products[index]
        if required_moments[target]:
            required_products[index] = True
            required_moments[left] = True
            required_moments[right] = True
    dead_products = sum(1 for required in required_products if not required)
    dead_nonbasic_moments = sum(
        1 for moment in range(basic_count, moment_count)
        if not required_moments[moment]
    )
    if dead_products or dead_nonbasic_moments:
        raise SystemExit(
            f"{path}: product graph contains unreachable work: "
            f"dead_products={dead_products} "
            f"dead_nonbasic_moments={dead_nonbasic_moments}"
        )
    print(
        f"{path}: reachable product graph OK "
        f"products={len(products)} scalars={len(scalars)}"
    )
PY

echo "SUS2-SH product graph reachability pruning check passed"
