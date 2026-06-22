#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./bin/mlp-sus2}
if [[ ! -x "$BIN" ]]; then
  echo "binary is not executable: $BIN" >&2
  exit 2
fi

tmp_dir=".codex_tmp/sh_direct_spatial_ace_init"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

"$BIN" check-sh-direct-spatial-ace-dev --lmax=4 >"$tmp_dir/direct_check.log"
grep -q "direct spatial ACE Gaunt check passed" "$tmp_dir/direct_check.log"
grep -q "odd_parity_cg_nonzero_gaunt_zero=" "$tmp_dir/direct_check.log"

common_opts=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --sh-factor-pruning=q-total
  --cutoff=7.5
  --radial-basis-size=4
  --radial-basis-type=RBChebyshev_sss
  --init-params=same
  --write-sh-scalar-info
)

"$BIN" init-sh "$tmp_dir/so3.mtp" "${common_opts[@]}" >/dev/null
"$BIN" init-sh "$tmp_dir/direct.mtp" "${common_opts[@]}" \
  --sh-coupling=direct-gaunt >/dev/null

if grep -q "sh_coupling" "$tmp_dir/so3.mtp"; then
  echo "default SO3 model should keep the historical model header without sh_coupling" >&2
  exit 1
fi
grep -q "sh_coupling = direct-gaunt" "$tmp_dir/direct.mtp"

python3 - "$tmp_dir/so3.mtp" "$tmp_dir/direct.mtp" <<'PY'
import re
import sys

so3 = open(sys.argv[1], "r", encoding="utf-8").read()
direct = open(sys.argv[2], "r", encoding="utf-8").read()

def int_field(text, name):
    match = re.search(rf"^{name}\s*=\s*(\d+)\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

so3_products = int_field(so3, "sh_product_count")
direct_products = int_field(direct, "sh_product_count")
if direct_products > so3_products:
    raise SystemExit(
        f"direct Gaunt product count should not exceed SO3-CG for this low-order model: "
        f"{direct_products} > {so3_products}"
    )
if so3 == direct:
    raise SystemExit("direct spatial ACE model unexpectedly matches SO3-CG model text")
print(
    "direct spatial ACE init metadata OK "
    f"so3_products={so3_products} direct_products={direct_products}"
)
PY

if [[ "${SUS2_SH_DIRECT_SPATIAL_ACE_SKIP_FD:-0}" != "1" ]]; then
  "$BIN" check-efs-fd-dev "$tmp_dir/direct.mtp" dev_test/NaCl_small.cfgs \
    --max-configs=1 \
    --max-atoms=2 \
    --displacement=1.0e-5 \
    --abs-tolerance=1.0e-4 \
    --rel-tolerance=1.0e-3 >"$tmp_dir/direct_fd.log"
  grep -q "force_components=6" "$tmp_dir/direct_fd.log"
  grep -q "stress_components=9" "$tmp_dir/direct_fd.log"
else
  echo "direct spatial ACE force/stress FD skipped by SUS2_SH_DIRECT_SPATIAL_ACE_SKIP_FD=1"
fi

echo "direct spatial ACE init check passed"
