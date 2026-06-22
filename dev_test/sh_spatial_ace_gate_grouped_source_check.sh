#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
hdr="$root/dev_src/mtpr.h"
cpp="$root/dev_src/mtpr_sh.cpp"

python3 - "$hdr" "$cpp" <<'PY'
import pathlib
import re
import sys

hdr = pathlib.Path(sys.argv[1]).read_text()
cpp = pathlib.Path(sys.argv[2]).read_text()

required_header = {
    "gate grouped derivative helper": "ApplySHStrictSpatialAceGateDers",
    "gate grouped tangent helper": "AccumulateSHStrictSpatialAceGateForward",
    "gate grouped mixed reverse helper": "AccumulateSHStrictSpatialAceGateMixedReverse",
    "gate grouped reverse helper": "BackpropSHStrictSpatialAceGate",
}
missing = [name for name, needle in required_header.items() if needle not in hdr]
if missing:
    raise SystemExit(
        "Missing strict spatial ACE gate grouped declarations: "
        + ", ".join(missing)
    )

def function_body(name):
    pattern = re.compile(
        r"\nvoid\s+MLMTPR::" + re.escape(name) + r"\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    match = pattern.search(cpp)
    if not match:
        raise SystemExit(f"missing function body: {name}")
    start = match.end()
    depth = 1
    i = start
    while i < len(cpp) and depth:
        if cpp[i] == "{":
            depth += 1
        elif cpp[i] == "}":
            depth -= 1
        i += 1
    if depth:
        raise SystemExit(f"unterminated function body: {name}")
    return cpp[start : i - 1]

required_calls = {
    "CalcTwoLayerGateScalarDers": [
        "ApplySHStrictSpatialAceGateDers(nbh)",
    ],
    "CalcTwoLayerGateWeightedScalarDers": [
        "BackpropSHStrictSpatialAceGate(site_energy_ders_wrt_moments_)",
    ],
    "CalcTwoLayerGateWeightedScalarDersForScalarSeeds": [
        "BackpropSHStrictSpatialAceGate(site_energy_ders_wrt_moments_)",
    ],
    "AccumulateTwoLayerGateScalarParamGrad": [
        "BackpropSHStrictSpatialAceGate(site_energy_ders_wrt_moments_)",
        "AccumulateSHStrictSpatialAceGateForward(",
        "AccumulateSHStrictSpatialAceGateMixedReverse(",
        "BackpropSHStrictSpatialAceGate(grad_dloss_dmom_)",
    ],
    "AccumulateTwoLayerGateScalarParamGradForScalarSeeds": [
        "BackpropSHStrictSpatialAceGate(site_energy_ders_wrt_moments_)",
        "BackpropSHStrictSpatialAceGate(sh_gate_energy_ders_wrt_moments_)",
        "AccumulateSHStrictSpatialAceGateForward(",
        "AccumulateSHStrictSpatialAceGateMixedReverse(",
        "BackpropSHStrictSpatialAceGate(grad_dloss_dmom_)",
    ],
    "CalcTwoLayerGateScalarDirectionalDerivatives": [
        "AccumulateSHStrictSpatialAceGateForward(",
    ],
}

missing = []
for function, needles in required_calls.items():
    body = function_body(function)
    for needle in needles:
        if needle not in body:
            missing.append(f"{function}: {needle}")

if missing:
    raise SystemExit(
        "Strict spatial ACE gate grouped path is incomplete:\n"
        + "\n".join(missing)
    )

print("strict spatial ACE gate grouped source check: PASS")
PY
