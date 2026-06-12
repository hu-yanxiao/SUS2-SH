# SUS2-SH Exact-Body Mu Gate

## Branch And Scope

This experiment lives on:

```text
codex/mu-body-order-gate
```

It is branched directly from SUS2-SH `origin/main`. The server work directory is:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex
```

The server binary is:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex/bin/mlp-sus2
```

The target smoke/test case directory is:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/
```

Submitted neighbor-site full test job in that directory:

```text
job id: 3791918
script: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/bsub_gate_3000_additive_mu_body_codex.lsf
log: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/mu_body_gate_3000_additive.log
```

Submitted double-site full test job in that directory:

```text
job id: 3792530
script: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/bsub_gate_3000_additive_mu_body_double_codex.lsf
log: /work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/mu_body_double_gate_3000_additive.log
status at submission: PEND
```

Double-site implementation commit hash:

```text
e9de9dd
```

## Formula

The gate signal is mu-dependent. For a channel \(\mu=(l,k)\), the scalar bucket is
selected by the exact body order:

\[
h_{j,\mu}
=
\sum_{r:\operatorname{body\_order}(s_{q_r})=k_\mu+1}
w_r s_{q_r}(j)
\]

The default neighbor-site SH moment is:

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
\left[1+A\tanh(a_{z_j,\mu}h_{j,\mu})\right]
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

The optional double-site gate first defines

\[
g_{a,\mu}=1+A\tanh(a_{z_a,\mu}h_{a,\mu})
\]

and then evaluates

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
g_{i,\mu}g_{j,\mu}
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

In code, \(k\) is zero-based, so the exact target body order is:

```text
k_internal + 2
```

For example, with `k-max=4`, channels with internal `k=0,1,2,3` use scalar
body orders `2,3,4,5`, respectively.

## CLI And Model Rules

- `--two-layer-gate-body-order` is not a normal option for this branch. Passing
  it explicitly is an error.
- `--two-layer-gate` requires:

  \[
  \texttt{body-order} \ge \texttt{k-max}+1
  \]

- `two_layer_gate_scalar_indices` automatically contains scalar basis functions
  with body orders `2..k_max+1`.
- `two_layer_gate_weights` remains one shared weight vector. The mu channel only
  selects the exact body-order bucket from that shared vector.
- The branch does not save or use `two_layer_gate_bias` / \(G_0\). Zero gate
  weights give \(h_{j,\mu}=0\), hence \(1+A\tanh(0)=1\), so the model reduces to
  ordinary SUS2-SH.
- New gate models write:

  ```text
  two_layer_gate_mode = mu-body-order
  two_layer_gate_site_mode = neighbor|double
  ```

  Legacy two-layer gate metadata is rejected instead of being silently read as
  the new mode.
- `--two-layer-residual` is not supported by this mu-body-order gate branch.

## Implementation Notes

- Gate caches are per atom times mu: \(h_{j,\mu}\), not one scalar per atom.
- Gate adjoints are also per atom times mu. During reverse mode, each mu
  adjoint only backpropagates into scalar weights whose exact body order matches
  `k_internal + 2`.
- The existing tanh cache, edge cache, site derivative cache, and shared-radial
  gate paths are preserved.
- CPU LAMMPS reads `two_layer_gate_mode = mu-body-order` and
  `two_layer_gate_site_mode`, builds the same exact body-order buckets,
  communicates per-atom per-mu gate values and adjoints, applies the same
  neighbor or double-site gate factor, and rejects legacy gate fields.
- Kokkos device gate evaluation is explicitly rejected for this branch rather
  than silently using the old scalar-gate kernels.

Interface verification:

```text
CPU LAMMPS pair source built into the independent branch LAMMPS work tree:
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps
Kokkos device pair style rejects mu-body-order gate models at settings time.
```

## Final Server Sync And Speed Decision

Final synchronized server binary:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex/bin/mlp-sus2
```

Final binary SHA-256:

```text
86eb63f7d3a8087e335ebe0d78e7681c1363649107ba63cd67f32b568e87ec6c
```

Independent CPU LAMMPS binary for this branch:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_double_avx2
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/bin/lmp.sus2_sh_cpu_avx2_mu_body_gate_opt_20260612
```

LAMMPS binary SHA-256:

```text
244007cfa6e5cb06d61a9c8f3d593dc20edc1fd9daca368453e9763b8c808d93
```

Final build and smoke evidence on the server:

```text
BUILD_RC=0
SMOKE_RC=0
```

The accepted optimization keeps the exact-body mu math but merges per-body-order
gate adjoints once per atom instead of rerunning the scalar reverse path once
for every body-order bucket. This preserves the existing tanh cache, edge
primitive cache, site-derivative cache, and shared-radial paths.

500x55 profile directory:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_mu_body_gate_opt1_speed_20260611_180753
```

Training-speed decision at matched profile settings:

| Version | Gradient total | Forward total | Decision |
| --- | ---: | ---: | --- |
| legacy scalar gate | `71.738 ms` | `61.323 ms` | old reference |
| accepted exact-body opt1 | `76.593 ms` | `62.535 ms` | synchronized |
| rejected final clear/cache candidate | `79.046 ms` | `62.561 ms` | rejected |

The final clear/cache candidate was stopped after collecting profile data
because it remained slower at `calls=570`. Its LSF job `3792234` therefore
ended with MPI interrupt text from the intentional stop, not from a correctness
failure.

## Double-Site Optimization Update 2026-06-12

The final double-site optimization keeps the existing exact-body gate math and
the existing neighbor-site fast paths. It adds a small center-gate scratch cache
for repeated `PrepareTwoLayerGatePairMuBuffers()` calls with the same center
atom and signal pointer, avoids computing `sech^2` when no derivative buffer is
requested, and avoids materializing the neighbor multiplier buffer unless the
caller asks for it. The LAMMPS CPU pair style also reuses per-pair mu adjoint
scratch arrays instead of allocating `std::vector<double>` inside the edge
adjoint loops.

Matched 500x55 profile directory:

```text
/work/phy-weigw/hyx/200w/5.31-new-44421/codex_mu_body_gate_double_opt_profile_20260612_082023
```

Final matched profile:

| Mode | Runtime | Linear total | Gradient total | Forward total |
| --- | ---: | ---: | ---: | ---: |
| neighbor-site | `478 s` | `428.333 ms` | `90.000 ms` | `68.500 ms` |
| double-site | `485 s` | `423.258 ms` | `96.217 ms` | `69.497 ms` |

The double-site runtime overhead at the matched profile settings is therefore
`485 / 478 = 1.015x`. Before this update the same comparison was `501 / 478 =
1.048x`, so the measured double-site overhead dropped from about `4.8%` to
about `1.5%`.

Fresh server verification after this optimization:

```text
mlp-sus2 build: PASS, SHA-256 86eb63f7d3a8087e335ebe0d78e7681c1363649107ba63cd67f32b568e87ec6c
sh_two_layer_gate_mu_body_order_check.sh: PASS
sh_two_layer_gate_double_mode_check.sh: PASS
sh_two_layer_gate_loss_gradient_check.sh: PASS
sh_two_layer_gate_force_fd_check.sh: PASS
LAMMPS neighbor additive smoke: PASS, max_abs=8.694700e-05
LAMMPS double additive smoke: PASS, max_abs=8.695400e-05
LAMMPS 1-rank vs 2-rank: PASS, max_abs=0
```

## Verification Checklist

Local serial checks:

```bash
bash dev_test/sh_two_layer_gate_mu_body_order_check.sh
bash dev_test/sh_two_layer_gate_double_mode_check.sh
bash dev_test/sh_two_layer_gate_init_check.sh
bash dev_test/sh_two_layer_gate_zero_compat_check.sh
bash dev_test/sh_two_layer_gate_forward_check.sh
bash dev_test/sh_two_layer_gate_force_fd_check.sh
bash dev_test/sh_two_layer_gate_loss_gradient_check.sh
bash dev_test/sh_two_layer_gate_shared_radial_check.sh
bash dev_test/sh_two_layer_gate_train_weight_check.sh
bash dev_test/sh_two_layer_gate_lmp_table_check.sh
bash dev_test/sh_plain_to_gate_upgrade_check.sh
LAMMPS_BIN=/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_double_avx2 \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
GATE_SITE_MODE=double \
LAMMPS_BIN=/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_double_avx2 \
  bash dev_test/lammps_two_layer_gate_additive_mlp_check.sh
```

Server checks:

- MPI `mlp-sus2` build in the independent work directory.
- `body-order < k-max+1` fails.
- Explicit `--two-layer-gate-body-order` fails.
- `k-max=4, body-order=6` gate scalars contain only body orders `2,3,4,5`.
- Single nonzero gate weights in body-order buckets `2,3,4,5` affect only their
  matching k channels.
- Zero gate weights match ordinary SH `calc-efs` energy and force.
- Double-site gate zero weights also match ordinary SH `calc-efs`, and nonzero
  double-site gate weights pass the loss-gradient check.
- CPU LAMMPS neighbor-site and double-site gate models match `mlp-sus2
  calc-efs` on the additive shared-radial smoke case. The 1-rank and 2-rank
  LAMMPS runs are bitwise identical in the checked energy/force values.
- Smoke run in `/work/phy-weigw/hyx/5.28-mof-cl-h2o/new-gate/` with the new
  independent binary.
