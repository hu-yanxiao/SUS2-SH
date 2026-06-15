# SUS2-SH Mu Body Linear-Combo Gate

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

## Server Binary Source Map

Current SUS2-SH main/developer reference binary for branch speed comparisons:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex/bin/mlp-sus2
```

Source relationship:

```text
server source tree: /work/phy-weigw/20260321_Test/SUS2-SH-work-codex
server checkout branch name: developer
GitHub repository: git@github.com:hu-yanxiao/SUS2-SH.git
GitHub branches at this commit: main, developer
commit: dd3010572b8d34fc53e7c371ea93ecc2104f74c1
binary SHA-256: aca08ae4d763c155af18dfa2c6a51b4c39379eef7712c29179d787ac4f499659
```

Current mu-body-order gate test binary:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex/bin/mlp-sus2
```

Source relationship:

```text
server source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/sus2-sh-developer-tanh
GitHub branch: codex/mu-body-order-gate
binary code commit: a827a02c13978fd7f06be757497d6b1cdb5bff9c
binary SHA-256: 0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
```

The server mu-body gate source mirror is not a git checkout. Treat the local
worktree and GitHub branch as the source authority, then sync confirmed files to
the server mirror before compiling. Documentation-only commits made after the
binary code commit do not imply that the binary must be rebuilt.

Do not use this historical directory as the current main reference:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-origin-main-speed-bench-codex
```

It was observed at detached commit `6a5ca72dd5a31766c0e0da0711a44682d8fa461c`,
older than the current main/developer reference, and produced an invalid speed
comparison because it rejected `RBLaguerre_log1p` during `init-sh`.

Fresh 40-rank CPU speed comparison on 2026-06-15:

```text
job id: 3797952
summary: /work/phy-weigw/hyx/xxx-b/test/cmp_20260615_093351_main_combo_full_summary.txt
train cfg: /work/phy-weigw/hyx/xxx-b/test/train.cfg
max-iter: 30
radial basis: RBLaguerre_log1p
```

| Case | Binary | Gate option | Runtime | Relative to main |
| --- | --- | --- | ---: | ---: |
| main/developer old gate | `SUS2-SH-work-codex/bin/mlp-sus2` | `--two-layer-gate-body-order=5` | `462 s` | `1.00x` |
| mu-body linear-combo | `SUS2-SH-mu-body-gate-work-codex/bin/mlp-sus2` | `--two-layer-gate-mode=mu-body-linear-combo` | `654 s` | `1.42x` |
| mu-scalar full | `SUS2-SH-mu-body-gate-work-codex/bin/mlp-sus2` | `--two-layer-gate-mode=mu-scalar-full` | `603 s` | `1.31x` |

All three runs exited with code 0.

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

The gate first forms one shared scalar per body order:

\[
H_{a,B}
=
\sum_{q:\operatorname{body\_order}(s_q)=B}
c_q s_q(a),
\qquad
B=2,\ldots,k_{\max}+1 .
\]

Each \(\mu\) channel then mixes those body-order scalars with its own
coefficients:

\[
h_{a,\mu}
=
\sum_{B=2}^{k_{\max}+1}
W_{\mu B}H_{a,B}.
\]

Equivalently,

\[
h_{a,\mu}
=
\sum_{B=2}^{k_{\max}+1}
W_{\mu B}
\sum_{q:\operatorname{body\_order}(s_q)=B}
c_q s_q(a).
\]

The additive tanh coefficient is species-only:

\[
a_{z,\mu}=a_z .
\]

The default neighbor-site SH moment is:

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
\left[1+A\tanh(a_{z_j}h_{j,\mu})\right]
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

The optional double-site gate first defines

\[
g_{a,\mu}=1+A\tanh(a_{z_a}h_{a,\mu})
\]

and then evaluates

\[
M_{i,\mu m}
=
\sum_{j\in N(i)} t_{z_i}t_{z_j}
g_{i,\mu}g_{j,\mu}
R_{\mu}(r_{ij})Y_{lm}(\hat r_{ij})
\]

Default initialization is:

\[
c_q^{(0)}=0,\qquad W_{\mu B}^{(0)}=1,\qquad a_z^{(0)}=1.
\]

Thus \(h_{a,\mu}^{(0)}=0\), the initial gate factor is exactly 1, and the
model starts from ordinary SUS2-SH while \(c_q\) still has a nonzero first-step
gradient.

## CLI And Model Rules

- `--two-layer-gate-body-order` is not a normal option for this branch. Passing
  it explicitly is an error.
- `--two-layer-gate` requires:

  \[
  \texttt{body-order} \ge \texttt{k-max}+1
  \]

- `two_layer_gate_scalar_indices` automatically contains scalar basis functions
  with body orders `2..k_max+1`.
- `two_layer_gate_weights` is the shared \(c_q\) vector and has length equal to
  `two_layer_gate_scalar_indices`.
- `two_layer_gate_body_mix_weights` is the \(W_{\mu B}\) matrix flattened as
  `radial_func_count * k_max`.
- The branch does not save or use `two_layer_gate_bias` / \(G_0\). Zero gate
  weights give \(h_{j,\mu}=0\), hence \(1+A\tanh(0)=1\), so the model reduces to
  ordinary SUS2-SH.
- New gate models write:

  ```text
  two_layer_gate_mode = mu-body-linear-combo|mu-scalar-full
  two_layer_gate_site_mode = neighbor|double
  ```

  Legacy two-layer gate metadata is rejected instead of being silently read as
  the new mode.
- `--two-layer-residual` is not supported by this mu-body-order gate branch.

## Implementation Notes

- Gate caches are per atom times mu: \(h_{j,\mu}\), not one scalar per atom.
- The implementation caches \(s_q\), compresses them to the small \(H_B\) vector,
  and then applies the \(W_{\mu B}\) matrix. This keeps the core signal build at
  \(O(N_q+N_\mu k_{\max})\) per atom instead of \(O(N_\mu N_q)\).
- Gate adjoints are also per atom times mu. During reverse mode, adjoints first
  accumulate into body-order buckets and then into the shared \(c_q\) scalar
  weights and scalar-value chain.
- The optional `mu-scalar-full` mode keeps one independent gate coefficient per
  \((\mu,q)\) pair while reusing the same scalar basis values \(s_q\). Its
  forward signal, force-chain scalar seeds, directional derivative, and gate
  weight gradient are batched through BLAS matrix/vector products instead of
  nested \(\mu,q\) loops.
- The existing tanh cache, edge cache, site derivative cache, and shared-radial
  gate paths are preserved.
- CPU LAMMPS reads `two_layer_gate_mode` and `two_layer_gate_site_mode`,
  supports both `mu-body-linear-combo` and `mu-scalar-full`, communicates
  compact per-body-order gate signals/adjoints for `mu-body-linear-combo`,
  communicates per-mu signals/adjoints for exact `mu-scalar-full`, applies the
  same neighbor or double-site gate factor, and rejects legacy gate fields.
- CPU LAMMPS now propagates gate force-chain derivatives by caching first-layer
  basic moment Jacobians per active edge and doing one combined scalar-product
  reverse pass per center atom, avoiding the old edge-by-scalar derivative
  materialization.
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
0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
```

Independent CPU LAMMPS binary for this branch:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_double_avx2
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/bin/lmp.sus2_sh_cpu_avx2_mu_body_gate_opt_20260612
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
```

LAMMPS binary SHA-256:

```text
7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
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

## Full-Mode And LAMMPS Efficiency Update

The retained `mu-scalar-full` mode is no longer a legacy fallback. It uses the
same scalar basis cache as `mu-body-linear-combo`, but stores an independent
flattened weight matrix over \((\mu,q)\). The hot training paths use BLAS for
the full weight projection, scalar-seed projection, directional derivative,
outer-product gate-weight gradient, and scalar-parameter seed accumulation.

Matched smoke-profile logs in `/work/phy-weigw/hyx/xxx-b/test`:

| Mode | Runtime | Forward total | Prepare gate | Main forward | Force chain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `262 s` | `165.620 ms` | `83.691 ms` | `58.805 ms` | `23.124 ms` |
| `mu-scalar-full` | `283 s` | `148.455 ms` | `78.817 ms` | `58.670 ms` | `10.968 ms` |

Fresh server verification after rebuilding current source:

```text
mlp-sus2 build: PASS, SHA-256 0f6bce2b6a15d97c1c848d4e1b247940fb0e961f1d104e4c661a35f7d595f315
CPU LAMMPS no-IPO build: PASS, SHA-256 7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
sh_two_layer_gate_mu_scalar_full_init_check.sh: PASS
sh_two_layer_gate_mu_scalar_full_loss_gradient_check.sh: PASS
sh_two_layer_gate_loss_gradient_check.sh: PASS
sh_two_layer_gate_force_fd_check.sh: PASS
LAMMPS mu-body-linear-combo smoke: PASS, max_abs=0
LAMMPS mu-body-linear-combo neighbor additive: PASS, max_abs=8.790000e-05
LAMMPS mu-body-linear-combo double additive: PASS, max_abs=8.788400e-05
LAMMPS mu-scalar-full smoke: PASS, max_abs=0
LAMMPS mu-scalar-full neighbor additive: PASS, max_abs=8.694700e-05
LAMMPS mu-scalar-full double additive: PASS, max_abs=8.695400e-05
LAMMPS 1-rank vs 2-rank: PASS, max_abs=0
```

## LAMMPS Gate-Mode Efficiency Update 2026-06-15

The independent CPU LAMMPS branch binary was rebuilt from the current
`codex/mu-body-order-gate` source after optimizing the exact gate dataflow:

```text
source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
binary SHA-256: 7c54d7ec5ee765d9ddf5fa2569628a84c004aac47ffbf161b6bc4b8e329ae4c7
pre-optimization backup: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.pre_body_comm_20260615_1237
```

Implementation notes:

- `mu-body-linear-combo` stores and communicates only the body-channel signals
  \(H_b\) and body-channel adjoints. Each receiving rank reconstructs
  \(h_\mu=\sum_b c_{\mu b}H_b\) locally before the tanh gate. During reverse
  communication, each per-mu gate adjoint is contracted back to
  \(dE/dH_b=\sum_\mu c_{\mu b}\,dE/dh_\mu\). This is algebraically exact and
  reduces the l4k4 communication width from 20 doubles to 4 doubles per atom.
- `mu-scalar-full` remains exact and still communicates the 20 per-mu
  \(h_\mu\) values/adjoints. The scalar basis is shared, and a scalar-major
  transposed weight view is used for the full-mode gate-force reverse
  contraction.
- The first-layer edge derivative cache now reserves once and writes by offset
  instead of appending large derivative slices with repeated vector `insert()`.
- LAMMPS gate communication pack/unpack uses exact block copies for contiguous
  gate-signal buffers; reverse unpack still accumulates per entry because LAMMPS
  reverse communication returns contributions to owned atoms.

Real B-system LAMMPS benchmark:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_130313
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
queue/cores: 1t88c, 96 MPI ranks
run: 5000 steps, tabstep 0.0005, trained models converted to *_lmp radial basis
main binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
new-branch binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
```

| Model/mode | Mean loop time | Relative to main |
| --- | ---: | ---: |
| main old gate | `4.339053 s` | `1.0000x` |
| `mu-body-linear-combo` | `5.337373 s` | `1.2301x` |
| `mu-scalar-full` | `6.796363 s` | `1.5663x` |

The first body-channel communication build on the same benchmark gave
`mu-body-linear-combo = 5.396523 s` (`1.2254x` against that run's main
baseline) and `mu-scalar-full = 6.999947 s` (`1.5894x`). The final build keeps
the full-mode scalar-major reverse optimization because it improves the full
absolute loop time, while the combo absolute time remains in the same range.

Numerical parity against the pre-body-channel-communication binary:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_125051
job: 3798420
combo abs_dE=0.000000e+00, max_force_diff=8.600000e-19, rms_force_diff=2.533799e-20
full  abs_dE=0.000000e+00, max_force_diff=3.903132e-15, rms_force_diff=6.774959e-16
```

Short 200-step profile on the same B 2x2x2 setup:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_125051
job: 3798421
```

| Mode | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `0.00116393` | `0.00028527` | `0.00034180` | `0.00043722` | `0.00046945` | `0.00016602` |
| `mu-scalar-full` | `0.00269330` | `0.00032071` | `0.00110001` | `0.00042119` | `0.00117981` | `0.00021778` |

The remaining `mu-scalar-full` gap is dominated by exact per-mu forward and
reverse communication. With arbitrary independent \(w_{\mu q}\), communicating
20 per-mu \(h_\mu\) values/adjoints is the compact exact representation; the
alternative of communicating all scalar basis values would be much larger.
Approximate low-rank compression is intentionally not used. The tested B-model
full weight matrix is numerically close to low-rank, but not exactly low-rank:
rank-1 SVD reconstruction leaves about `1e-4` max absolute residual and rank-7
still leaves about `3e-10`. The body-mix matrix is also only numerically close
to rank-1, with about `1.7e-10` rank-1 residual. These are approximation
opportunities, not exact algorithmic reductions, and are therefore out of scope.

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
