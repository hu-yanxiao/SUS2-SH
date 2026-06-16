#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cpp="$root/interfaces/lammps/ML-SUS2/pair_sus2_mtp.cpp"
hdr="$root/interfaces/lammps/ML-SUS2/pair_sus2_mtp.h"

python3 - "$cpp" "$hdr" <<'PY'
import pathlib
import sys

cpp = pathlib.Path(sys.argv[1]).read_text()
hdr = pathlib.Path(sys.argv[2]).read_text()

required = {
    "header sparse schedule": "two_layer_gate_product_indices",
    "forward sparse products": "forward_two_layer_gate_products",
    "reverse sparse products": "backprop_two_layer_gate_products",
    "schedule push": "two_layer_gate_product_indices.push_back",
}

missing = [
    name
    for name, needle in required.items()
    if needle not in (hdr if name.startswith("header") else cpp)
]
if missing:
    raise SystemExit(
        "Missing sparse two-layer-gate product schedule support: "
        + ", ".join(missing)
    )

if "for (int k = 0; k < gate_product_limit; k++)" in cpp:
    raise SystemExit(
        "Gate forward path still uses a prefix product loop instead of the sparse schedule."
    )

if "backprop_sh_products_from(gate_moments, gate_product_limit)" in cpp:
    raise SystemExit(
        "Gate reverse path still uses prefix product backprop instead of the sparse schedule."
    )

print("sparse two-layer-gate product schedule source check: PASS")
PY
