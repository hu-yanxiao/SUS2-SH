#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_residual_forward"
mkdir -p "$tmp_dir"

plain="$tmp_dir/plain.mtp"
residual="$tmp_dir/residual.mtp"
plain_pred="$tmp_dir/plain_pred.cfg"
residual_pred="$tmp_dir/residual_pred.cfg"
train="$tmp_dir/one.cfg"
rm -f "$plain" "$residual" "$plain_pred" "$residual_pred" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

common_opts=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --write-sh-scalar-info
)

./bin/mlp-sus2 init-sh "$plain" "${common_opts[@]}"
./bin/mlp-sus2 init-sh "$residual" "${common_opts[@]}" \
  --two-layer-gate \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-shared-radial \
  --two-layer-residual

python3 - "$plain" "$residual" <<'PY'
import re
import sys

for path in sys.argv[1:]:
    text = open(path).read()
    alpha_match = re.search(r"alpha_scalar_moments = ([0-9]+)", text)
    if not alpha_match:
        raise SystemExit(f"missing alpha_scalar_moments in {path}")
    alpha_count = int(alpha_match.group(1))

    def replace_or_append_list(name, values):
        nonlocal_text[0] = nonlocal_text[0].rstrip()
        replacement = name + " = {" + ", ".join(f"{x:.15e}" for x in values) + "}"
        match = re.search(rf"{name} = \{{([^}}]*)\}}", nonlocal_text[0])
        if match:
            nonlocal_text[0] = (
                nonlocal_text[0][:match.start()]
                + replacement
                + nonlocal_text[0][match.end():]
            )
        else:
            nonlocal_text[0] += "\n" + replacement + "\n"

    nonlocal_text = [text]
    replace_or_append_list("species_coeffs", [1.0, 1.0])
    replace_or_append_list("moment_coeffs", [1.0e-3] * alpha_count)
    open(path, "w").write(nonlocal_text[0])
PY

./bin/mlp-sus2 calc-efs "$plain" "$train" "$plain_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$residual" "$train" "$residual_pred" >/dev/null

python3 - "$plain_pred" "$residual_pred" <<'PY'
import sys

def read_values(path):
    values = []
    force_columns = None
    expect_energy = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if expect_energy:
                values.append(float(line.split()[0]))
                expect_energy = False
            elif line == "Energy":
                expect_energy = True
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                force_columns = [header.index(name) for name in ("fx", "fy", "fz")]
            elif force_columns is not None and line[0].isdigit():
                parts = line.split()
                if len(parts) > max(force_columns):
                    values.extend(float(parts[idx]) for idx in force_columns)
    return values

plain = read_values(sys.argv[1])
residual = read_values(sys.argv[2])
if len(plain) != len(residual):
    raise SystemExit(f"value count mismatch: {len(plain)} vs {len(residual)}")
err = max((abs(a - b) for a, b in zip(plain, residual)), default=0.0)
if err >= 1.0e-10:
    raise SystemExit(f"zero residual/direct-gate predictions differ from plain SH: {err}")
PY
