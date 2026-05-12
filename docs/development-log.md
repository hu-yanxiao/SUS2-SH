# SUS2-SH Development Log

## 2026-05-12 Initial C++ Prototype

User decisions:

- Use SH basic moments `B[k,l,m] = sum_j R_{k,l}(r_ij) Y_lm(rhat_ij)`.
- Use O(3)-even scalar features only.
- Use canonical leaf ordering by full `(l,k)` pair, not separate `l` and `k`
  sorting.
- Interpret `body_order = 5` as up to four SH factors.
- Use `l3k3`, cutoff `7.5`, `radial_basis_size = 10`, and
  `radial_basis_type = RBChebyshev_sss` for the first Cu-Zr smoke training.
- Keep GPUMD as a documented plan for now; first priority is training-side C++.
- Production training logic must remain the original SUS2-MLIP logic. SUS2-SH
  changes the angular basis and contraction/chain-rule implementation, not the
  outer trainer semantics.

Prototype scope:

- C++17 standalone executable `sus2-sh`.
- MLIP CFG reader with periodic image neighbor enumeration.
- Complex SH basis up to `l=3`.
- Clebsch-Gordan coefficients from Wigner-3j formula.
- Standard-tree scalar contractions through four SH factors.
- Energy-only normalized LMS training and JSON model output.

Important limitation:

- This first prototype keeps all standard-tree intermediate `L` channels for
  four-factor terms. For `l3k3 body_order=5`, that gives 1409 scalar features
  without bias. A symmetry-pruned independent basis should be added later if we
  want the smaller 1103-count basis discussed during planning.
- The standalone `train` command is a smoke harness. It is not the final trainer
  architecture. The production implementation should reuse the existing
  SUS2-MLIP C++ model/trainer interfaces and replace only the angular basis,
  scalar generation, and reverse-mode derivative path.
- The production BFGS path still needs the SH equivalent of
  `AccumulateCombinationGrad`, including analytic gradients with respect to
  radial coefficients and scaling parameters. Linear normal equations are only a
  check around the linear coefficient surface.

## Server Smoke Install

Server install path:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex
```

Cu-Zr test data:

```text
/work/phy-weigw/hyx/cu-zr/sus2-sh/train.cfg
```

Build command:

```bash
cd /work/phy-weigw/20260321_Test/SUS2-SH-work-codex
make
```

Topology check:

```text
l_max=3 k_max=3 body_order=5 max_factors=4
channels=12
scalar_count_without_bias=1409
scalar_count_with_bias=1410
order_1_factor_scalars=3
order_2_factor_scalars=24
order_3_factor_scalars=137
order_4_factor_scalars=1245
```

Force chain-rule finite-difference check:

```text
energy=125115.16534723347
force_fd_check_components=6
force_fd_rms_abs_error=3.1719461587596148e-05
force_fd_max_abs_error=4.7713710188190817e-05
```

Energy-only smoke training:

```text
loaded_cfgs=64
feature_count=1410
radial_basis_type=RBChebyshev_sss radial_basis_size=10 cutoff=7.5
final_train_energy_mae_eV_per_atom=3.75117e-05
wrote_model=/work/phy-weigw/20260321_Test/SUS2-SH-work-codex/sus2-sh-l3k3-b5-smoke.model.json
```

Notes:

- A Wigner-3j factorial bug was found by the first chain-rule check. The
  four-factor path can use intermediate `L=6`, so factorial support must exceed
  12. The prototype now uses `tgamma`.

## E/F/S Harness Update

The smoke harness now includes:

- energy rows from scalar features;
- force rows from the same reverse-mode SH chain rule;
- stress rows using the SUS2 sign convention
  `stress[a][b] -= dE_site/d r_ij[a] * r_ij[b]`.

The Cu-Zr file at `/work/phy-weigw/hyx/cu-zr/sus2-sh/train.cfg` has no
`PlusStress`, `Stress`, or `Virial` blocks, so `stress_rows=0` for that data.

Queue-33 validation job:

```text
job_id = 3657525
queue = 33
workdir = /work/phy-weigw/20260321_Test/SUS2-SH-work-codex
script = scripts/cuzr_l3k3_b5_efs_33.lsf
```
