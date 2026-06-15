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

Current optimized LAMMPS mu-body gate binary:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
```

Source relationship as of 2026-06-16 03:00 CST:

```text
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/sus2-sh-developer-tanh
local branch: codex/mu-body-order-gate
GitHub/code commit: a8ec8a98b41e99f12821567d61f5cd8b602c4221
server SUS2-SH source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex/interfaces/lammps/ML-SUS2
server LAMMPS build src: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
LAMMPS binary SHA-256: e77ee6eaa74ef6eb9ad8b4723dc649be907856deb881ef734981cd77cd864054
pair_sus2_mtp.cpp SHA-256: 51a09e0d403af744b2e05770a45021b4adc9cbc508ddb245c9ff3041b67fea80
pair_sus2_mtp.h SHA-256: 6bc7a1624aae0b0f162f02bb4fcc77f983a8700a4960f81f78ad1f040918bd55
```

The stable binary above is the same file content as:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.no_center_branchless_trial
```

Final exact LAMMPS optimization status:

```text
test directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial
main LAMMPS baseline binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
40-rank parity job for final SHA-256 e77...4054: 3801127, DONE
40-rank performance job for final SHA-256 e77...4054: 3801128, DONE
```

Verified 40-rank parity for the final binary:

```text
combo abs_dE=0, max_force_diff=0
full abs_dE=0, max_force_diff=0
```

Final 40-rank LAMMPS speed comparison:

```text
summary: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/no_center_branchless_perf40_summary.txt
host: b07u32a
replicate: 2 2 2
run steps: 5000
atoms after replicate: 384
```

| Case | Mean Loop Time | Relative To Main |
| --- | ---: | ---: |
| main old gate | `14.760067 s` | `1.0000x` |
| mu-body linear-combo | `16.754067 s` | `1.1351x` |
| mu-scalar full | `17.457800 s` | `1.1828x` |

Both active mu gate modes are within the target `1.2x` main-branch LAMMPS
speed envelope on this benchmark. The exact optimizations promoted in the
final binary are:

- mode-specific gate-force cache strategy: body-linear-combo uses cached basic
  Jacobians; mu-scalar-full caches radial values/derivatives and recomputes the
  compact SH Jacobian on demand;
- grouped \(\mu\)-major SH Jacobian contraction for the on-demand full mode;
- direct stride-4 body-linear-combo gate-signal and adjoint contractions;
- branchless full-mode on-demand contraction over grouped SH basic terms;
- single-gate no-center main-layer force/adjoint path, leaving the double-gate
  center path separate.

Current LAMMPS mu-body gate trial binary for exact gate performance work:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.fulldedupe_trial
```

Source relationship as of 2026-06-15 21:16 CST:

```text
local source worktree: /Users/hu-yanxiao/Projects/SUS2MLIP/.codex_tmp/sus2-sh-developer-tanh
local branch: codex/mu-body-order-gate
GitHub commit: c28fe23
server SUS2-SH source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex/interfaces/lammps/ML-SUS2
server LAMMPS build src: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
LAMMPS trial binary SHA-256: e2f8f2d6e23eed2196f28c207fd2b2da9ef96022943853dcdcc7555dae42c270
pair_sus2_mtp.cpp SHA-256: b5ed9f68d797655d581d1e969390e29ce1f925bef0f2bcdafac1f8a32db06872
pair_sus2_mtp.h SHA-256: 8d8e1c3e68b545f3e211e6accdbc2aaf29356cbf66333afc3467675680809ebd
```

The interface source hashes above were checked to match between the local
worktree, the server SUS2-SH source mirror, and the LAMMPS build source tree.

Current exact LAMMPS optimization status for this trial:

```text
test directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial
main LAMMPS baseline binary: /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
40-rank parity job for trial SHA-256 e2f8...c270: 3800824, DONE
40-rank speed job for trial SHA-256 e2f8...c270: 3800825, DONE
96-rank parity job for trial SHA-256 e2f8...c270: 3800826, submitted
96-rank speed job for trial SHA-256 e2f8...c270: 3800827, submitted
```

Verified login-node and 40-rank parity for the current trial:

```text
combo abs_dE=0, max_force_diff=0
full login max_force_diff=6.634440e-15
full 40-rank max_force_diff=6.689640e-15
```

Clean 40-rank LAMMPS speed comparison for trial SHA-256 `e2f8...c270`:

```text
summary: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/speed_fulldedupe_speed40_summary.txt
host: b08u22a
replicate: 2 2 2
run steps: 5000
atoms after replicate: 384
```

| Case | Mean Loop Time | Relative To Main | Relative To Accepted |
| --- | ---: | ---: | ---: |
| main old gate | `11.872133 s` | `1.0000x` | - |
| accepted mu-body linear-combo | `17.081967 s` | `1.4388x` | `1.0000x` |
| e2f8 mu-body linear-combo | `16.299800 s` | `1.3729x` | `0.9542x` |
| accepted mu-scalar full | `18.118667 s` | `1.5262x` | `1.0000x` |
| e2f8 mu-scalar full | `17.290133 s` | `1.4564x` | `0.9543x` |

Short 40-rank profile for trial SHA-256 `e2f8...c270`:

```text
job id: 3800839
summary: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/profile40_e2f8_summary.txt
host: b08u03a
profile steps: 20
```

| Mode | Mean Total | First Layer | Forward Comm | Main Layer | Reverse Comm | Gate Force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mu-body linear-combo | `0.003324401 s` | `0.001049122 s` | `0.000482758 s` | `0.001607793 s` | `0.000611042 s` | `0.000525844 s` |
| mu-scalar full | `0.003486166 s` | `0.001122569 s` | `0.000550462 s` | `0.001393776 s` | `0.000564622 s` | `0.000735240 s` |

The stage timings are reduced by MPI max per stage, so the component means do
not need to sum exactly to the total. The actionable signal is that full mode
still spends more in gate-force adjoint work, while communication is already a
large fraction of the 40-rank short-run profile.

Rejected exact LAMMPS signal-derivative-cache candidate:

```text
candidate binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.signalderiv_trial
candidate SHA-256 after alias fix: ba05caca66e9690b15b189acc6c63449aa1db4df8e5e114526e10cc0cd4c3c34
login parity: combo max_force_diff=1.0e-20, full max_force_diff=1.3878e-16
login profile combo mean_total=0.087373956 s
login profile full mean_total=0.219957343 s
decision: rejected; exact but much slower than e2f8 due 4/20-RHS product-graph reverse work moved into the first layer
```

This candidate precomputed \(\partial h_{i,s}/\partial r_{ij}\) for every gate
signal \(s\) and edge, reducing the late gate-force cache from
`alpha_index_basic_count` entries per edge to `gate_signal_stride` entries per
edge. The math is exact after explicitly handling product-graph square terms
where `a0 == a1`, but the extra multi-RHS scalar-product reverse pass dominates
the saved memory traffic on the tested model. Do not promote this candidate.

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

### No-Zero Gate-Value LAMMPS Update 2026-06-15

The current accepted CPU LAMMPS branch binary adds two exact dataflow
optimizations on top of the body-channel communication build:

- The first-layer gate moment cache is reused as a member buffer instead of
  reallocated every step, and the scalar scratch buffer is resized once.
- `mu-scalar-full` accumulates the full gate projection through a scalar-major
  transposed weight view so each scalar value is loaded once and the 20
  \(\mu\)-weights are contiguous.
- `two_layer_gate_values` is no longer pre-zeroed over all local and ghost atoms
  before the gate forward pass. Each local atom's gate signal is fully written by
  the first-layer pass, static fixed atoms are copied from their cache, and ghost
  signals are fully written by LAMMPS forward communication. The adjoint buffer
  is still zeroed because reverse communication accumulates into it.

Accepted independent CPU LAMMPS binary:

```text
binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
binary SHA-256: 9e89565433abd450bf49ecb85e702b84f6bf290614c6b499d3ec353e9cae83d9
source code commit: f8a098f643eecfdfe9b9b2e0fc144d8046518083
previous backup: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.pre_nozero_values_20260615_1515
trial binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.nozero_values_trial
source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
SUS2-SH server mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex
```

Numerical parity against the previous canonical branch binary:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_gate_nozero_values_20260615_1515
job: 3798701
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=5.922340e-15, rms_force_diff=1.163142e-15
```

Real B-system LAMMPS speed comparison in one matched 96-rank job:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_gate_nozero_values_20260615_1515
job: 3798702
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
run: 5000 steps, tabstep 0.0005, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main |
| --- | ---: | ---: |
| main old gate | `4.514740 s` | `1.0000x` |
| canonical `mu-body-linear-combo` before this update | `5.518320 s` | `1.2223x` |
| cache/scalar-major `mu-body-linear-combo` trial | `5.479610 s` | `1.2137x` |
| accepted no-zero `mu-body-linear-combo` | `5.434567 s` | `1.2037x` |
| canonical `mu-scalar-full` before this update | `6.908270 s` | `1.5302x` |
| cache/scalar-major `mu-scalar-full` trial | `6.894727 s` | `1.5272x` |
| accepted no-zero `mu-scalar-full` | `6.755803 s` | `1.4964x` |

Short 200-step profile with the accepted binary:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_gate_nozero_values_20260615_1515
job: 3798812
```

| Mode | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu-body-linear-combo` | `0.00122208` | `0.00027351` | `0.00034509` | `0.00042426` | `0.00056258` | `0.00015259` |
| `mu-scalar-full` | `0.00310239` | `0.00037310` | `0.00140737` | `0.00045936` | `0.00150011` | `0.00023858` |

The no-zero update gives a small exact improvement, but the same conclusion
holds: on this very small 384-atom/96-rank benchmark, the remaining full-mode
gap is communication dominated. `mu-body-linear-combo` must communicate 4
body-channel signals/adjoints for the l4k4 model; `mu-scalar-full` must
communicate 20 independent per-\(\mu\) signals/adjoints. The LAMMPS pack/unpack
path already sends contiguous signal blocks. Reducing the full-mode width below
20 would require an exact low-rank/sparse identity in the trained weight matrix
or a change in the mathematical model; no approximate compression is used.

### Raw Edge/Comm Direct LAMMPS Update 2026-06-15

The current accepted CPU LAMMPS branch binary adds exact data-layout and hot-loop
optimizations on top of the no-zero build:

- Gate edge staging uses LAMMPS-managed raw contiguous buffers instead of
  per-step `std::vector` clear/reserve/push/resize traffic.
- The common l4k4 `mu-body-linear-combo` path specializes the four body-channel
  signal and adjoint maps while keeping the same per-\(\mu\) accumulation order.
- Forward and reverse gate communication add fixed-width stride 4 and stride 20
  pack/unpack paths. The communicated values and adjoint sums are unchanged.

Accepted independent CPU LAMMPS binary:

```text
binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo
binary SHA-256: 89e29919ddf899990f5f8e4e1e0a55be5ce0a31bc6bc04c0acb038bc795714ad
source code commit: 560558dee1f1f2bb38b7b815039256d669d1984e
previous backup: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.pre_raw_edge_comm_direct_pack_20260615_1900
trial binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.raw_edge_comm_direct_pack_trial
source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
SUS2-SH server mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex
```

Numerical parity against the previous canonical branch binary:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial
job: 3800591
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
```

Real B-system LAMMPS speed comparison in one matched 96-rank job:

```text
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial
job: 3800594
node: be1u39a
data: first structure from /work/phy-weigw/hyx/xxx-b/test/train.cfg
LAMMPS cell: replicate 2 2 2, 384 B atoms
run: 5000 steps, tabstep 0.0005, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main | Relative to previous branch binary |
| --- | ---: | ---: | ---: |
| main old gate | `4.336693 s` | `1.0000x` | n/a |
| previous `mu-body-linear-combo` | `5.574307 s` | `1.2854x` | `1.0000x` |
| accepted raw-edge/comm-direct `mu-body-linear-combo` | `5.321253 s` | `1.2270x` | `0.9546x` |
| previous `mu-scalar-full` | `6.871523 s` | `1.5845x` | `1.0000x` |
| accepted raw-edge/comm-direct `mu-scalar-full` | `6.809597 s` | `1.5702x` | `0.9910x` |

An earlier same-node full-speed run for the direct version without the
forward-pack specialization gave `mu-body-linear-combo = 5.313430 s`
(`1.1989x` relative to that job's main run) and `mu-scalar-full = 6.914083 s`.
Across the matched jobs, the accepted code gives a repeatable improvement over
the previous branch binary. The remaining `mu-scalar-full` gap is still
communication dominated because this mode exactly carries 20 independent
per-\(\mu\) gate signals and adjoints for the l4k4 model.

### Direct Derivative-Flow LAMMPS Update 2026-06-15

The current retained CPU LAMMPS branch source adds another exact memory-flow
cleanup on top of the raw-edge/comm-direct build:

- The first-layer gate SH pass writes the raw edge derivative planes directly
  into the gate derivative cache, instead of first writing the ordinary
  `moment_jacobian_{x,y,z}` slabs and then copying them.
- The first-layer gate moment buffer is cleared only through the gate scalar
  dependency width, not the full model moment width.
- The late gate-force scalar seed is accumulated directly into
  `nbh_energy_ders_wrt_moments`; the temporary scalar scratch vector is removed.

Retained independent CPU LAMMPS binary:

```text
binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.directfill_seed_trial
binary SHA-256: 08b53882086b4e99eabfbbed60a84f5b00932d0ac33a9a878d1d3a754c3ac0f8
source code commit: 826415d4dcc64ee0a38ba445c2302d6f7aa0d5ec
source mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/lammps/src
SUS2-SH server mirror: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-work-codex
pair_sus2_mtp.cpp SHA-256: 6f80c0d1a1df873806050e02bf7f2c9c9d0e16b73d19abf288d76b4b5a2ae375
pair_sus2_mtp.h SHA-256: db75fb2da94ef56eed5d19e9260fddd589f39d1e8402fae74c878ddc582ea810
```

Current login-node parity against the previous `e2f8` canonical branch binary:

```text
script: /work/phy-weigw/20260321_Test/run_directfill_seed_login_parity.sh
combo abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
full  abs_dE=0.000000e+00, max_force_diff=0.000000e+00, rms_force_diff=0.000000e+00
```

Current login-node short profile against the previous `e2f8` canonical branch
binary:

```text
script: /work/phy-weigw/20260321_Test/run_directfill_seed_login_profile.sh
directory: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/directfill_seed_login_profile
```

| Mode/binary | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2f8` `mu-body-linear-combo` | `0.083261775` | `0.029778039` | `0.000018864` | `0.038592064` | `0.000013142` | `0.014833429` |
| retained `mu-body-linear-combo` | `0.079464204` | `0.025972551` | `0.000019252` | `0.038679035` | `0.000014232` | `0.014754995` |
| `e2f8` `mu-scalar-full` | `0.088467262` | `0.031465805` | `0.000089547` | `0.037235977` | `0.000054461` | `0.019555161` |
| retained `mu-scalar-full` | `0.082831310` | `0.026925359` | `0.000088189` | `0.036981662` | `0.000055227` | `0.018714115` |

The login-node profile is only a hot-path direction check; the real LAMMPS
speed comparison remains the submitted 40-rank/96-rank B-system jobs in the same
test directory.

Completed 40-rank short profile on the B-system 2x2x2 LAMMPS case:

```text
job: 3800991
node: b07u38a
summary: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/seed_profile40_summary.txt
```

| Mode/binary | total | first | fcomm | main | rcomm | gate_force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2f8` `mu-body-linear-combo` | `0.003463722` | `0.001082097` | `0.000418600` | `0.001682915` | `0.000592926` | `0.000633548` |
| retained `mu-body-linear-combo` | `0.003338736` | `0.000965856` | `0.000370418` | `0.001693265` | `0.000599926` | `0.000610887` |
| `e2f8` `mu-scalar-full` | `0.003881440` | `0.001175615` | `0.000508676` | `0.001652033` | `0.000620843` | `0.000888017` |
| retained `mu-scalar-full` | `0.003810493` | `0.001045825` | `0.000467634` | `0.001706453` | `0.000669589` | `0.000885084` |

Completed 40-rank real LAMMPS speed comparison on the B-system 2x2x2 case:

```text
job: 3801017
node: b07u38a
summary: /work/phy-weigw/hyx/xxx-b/test/codex_b_cfg_trained_lammps_perf_20260615_raw_edge_trial/speed_seed_ipo40_summary.txt
run: 5000 steps, 384 B atoms, trained *_lmp models
```

| Model/mode | Mean loop time | Relative to main | Relative to e2f8 |
| --- | ---: | ---: | ---: |
| main old gate | `14.571533 s` | `1.0000x` | n/a |
| `e2f8` `mu-body-linear-combo` | `17.898400 s` | `1.2283x` | `1.0000x` |
| retained no-IPO `mu-body-linear-combo` | `17.440233 s` | `1.1969x` | `0.9744x` |
| retained IPO `mu-body-linear-combo` | `17.425167 s` | `1.1958x` | `0.9736x` |
| `e2f8` `mu-scalar-full` | `19.782833 s` | `1.3576x` | `1.0000x` |
| retained no-IPO `mu-scalar-full` | `19.227633 s` | `1.3195x` | `0.9719x` |
| retained IPO `mu-scalar-full` | `19.432767 s` | `1.3336x` | `0.9823x` |

The retained no-IPO build is the best full-mode result in this matched job, and
IPO is neutral/slightly better only for combo. The direct derivative-flow
cleanup therefore remains useful but not sufficient to make either new gate
mode reach the old main gate speed on this small 40-rank B benchmark.

At the time this note was updated, the remaining pending job was:

```text
96-rank speed: 3801039
```

Rejected exact fast-dispatch candidate:

```text
trial binary: /work/phy-weigw/20260321_Test/SUS2-SH-mu-body-gate-lammps-work-codex/bin/lmp.ml-sus2_mu_body_gate_avx2_noipo.fastdispatch_trial
trial SHA-256: 67b4c698d963eec3f9e1f48542bd067eb5c82530615d8d0a9a1dc2de204d6e34
login parity: combo/full zero energy and force differences against the retained binary
login profile: combo total 0.079477085 -> 0.079802535, full total 0.083516155 -> 0.083533995
decision: rejected; exact but no measured speed gain.
```

The benchmark models were also checked for exact structural compression
opportunities. The trained `mu-body-linear-combo` body-mix rows and
`mu-scalar-full` rows are numerically close to each other, but they are not
exactly duplicate rows. Therefore the communication width cannot be reduced
without changing the mathematical model or using approximation, which is out of
scope for this branch.

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
