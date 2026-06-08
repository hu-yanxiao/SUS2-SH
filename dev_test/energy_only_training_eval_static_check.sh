#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$ROOT/dev_src/non_linear_regression.cpp" "$ROOT/dev_src/non_linear_regression.h" <<'PY'
import re
import sys
from pathlib import Path

cpp_path = Path(sys.argv[1])
header_path = Path(sys.argv[2])
cpp = cpp_path.read_text()
header = header_path.read_text()

def function_body(name):
    signature = (
        rf"void\s+NonLinearRegression::{name}\s*"
        rf"\(\s*const\s+Configuration\s*&\s*orig\s*,\s*"
        rf"const\s+Neighborhoods\s*\*\s*neighborhoods\s*\)\s*\{{"
    )
    match = re.search(signature, cpp)
    if not match:
        raise SystemExit(f"missing neighborhood NonLinearRegression::{name}")
    depth = 0
    start = match.end() - 1
    for index in range(start, len(cpp)):
        char = cpp[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cpp[start:index + 1]
    raise SystemExit(f"unterminated NonLinearRegression::{name}")

if "EvaluateTrainingConfiguration" not in header:
    raise SystemExit("NonLinearRegression must route loss evaluation through EvaluateTrainingConfiguration")
if "AccumulateEnergyCombinationGrad" not in cpp:
    raise SystemExit("energy-only training gradients must use AccumulateEnergyCombinationGrad")

for name in ("AddLoss", "AddLossGrad"):
    body = function_body(name)
    if "CalcEFS(" in body:
        raise SystemExit(f"{name} still calls CalcEFS directly")

print("energy_only_training_eval_static_check passed")
PY
