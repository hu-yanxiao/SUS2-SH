# Two-layer gate shared-radial optimization notes (2026-06-02)

Repository branch: `developer`

Server tree:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex
```

Server binary after the kept changes:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex/bin/mlp-sus2
sha256: 5ae762967ca2104cea95b48daffca67f2b110216bcc42bfcd95ae52e007a9bdf
```

## Kept changes

The kept changes only affect the two-layer gate evaluation path.

- Added two-level directed-edge primitive cache capability:
  - full cache for parameter-gradient paths,
  - coordinate-only cache for forward/linear component paths.
- Reused the directed-edge primitive cache in active two-layer `CalcSHBasisFuncsDers`.
- Replaced the temporary `gate_basis_ders` matrix in linear component building with direct accumulation into per-atom gate linear adjoints.
- Skipped full `basis_ders` zeroing only when evaluating an active two-layer gated basis; all scalar derivative rows are overwritten in that path.

## Rejected attempts

These were compiled/tested but not kept.

- Fusing gate JVP into `CalcSHBasisFuncsDers` increased `main_components` and total linear build time.
- Forcing product-row evaluation in active gated basis was mathematically equivalent but slower on the 500-sample benchmark.
- Directly accumulating terminal scalar gate adjoints avoided one intermediate store but increased `gate_adjoints` time from about 90 ms to about 130 ms.

## Verification

Server smoke/FD checks passed:

```text
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_residual_force_fd_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_residual_loss_gradient_check.sh
bash .codex_tmp/shared_radial_loss_gradient_check.sh
./bin/mlp-sus2 check-linear-components-fd-dev \
  .codex_tmp/sh_two_layer_gate_shared_radial_loss_gradient/shared_nonzero.mtp \
  .codex_tmp/sh_two_layer_gate_shared_radial_loss_gradient/one.cfg \
  --max-configs=1 --max-atoms=1 --displacement=1.0e-5 \
  --abs-tolerance=5.0e-5 --rel-tolerance=5.0e-4
```

The shared-radial non-residual loss-gradient check covered the gate-radial
coefficient segment:

```text
checked_coeffs=154
gate_radial_begin=86
gate_radial_end=110
abs_err=0
rel_err=0
```

The shared-radial linear-component FD check produced:

```text
checked_components=99
coeff_count=33
abs_err=1.11022310324e-11
```

## Performance evidence

500-sample active two-layer shared-radial profile command was based on:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_two_layer_gate_shared_radial_profile_active_500x55_linearcache_basisreuse_20260602_045923
```

Clean reference:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_two_layer_gate_shared_radial_profile_active_500x55_clean_20260602_0405
avg_us_total ~= 412635
avg_us_main_components ~= 229895
avg_us_gate_adjoints ~= 90148
TrainLinear build_s at step 50 ~= 2.148 s
```

Kept direct/cache/basis-reuse profile:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_two_layer_gate_shared_radial_profile_active_500x55_linearcache_basisreuse_20260602_045923
avg_us_total ~= 375481
avg_us_main_components ~= 201274
avg_us_gate_adjoints ~= 90128
TrainLinear build_s at step 50 ~= 1.976 s
```

Kept skip-basis-zero profile:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_two_layer_gate_shared_radial_profile_active_500x55_skip_basis_zero_20260602_053729
avg_us_total ~= 369464
avg_us_main_components ~= 195490
avg_us_gate_adjoints ~= 90190
TrainLinear build_s at step 50 ~= 1.952 s
```

Net observed linear component build improvement versus the clean reference was
about 10% on this 500-sample, 120-rank profile.
