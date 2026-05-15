# SUS2-SH Development Notes

## Current implementation target

SUS2-SH is developed by copying the SUS2-MLIP developer version and replacing the angular expression with real spherical harmonics. The original training flow remains the driver: `mlp-sus2 train`, E/F/S equations, nonlinear BFGS, linear sub-solve, model save/load, and MPI execution stay in the SUS2 code path.

## Training model

The training `.mtp` keeps the ordinary SUS2 radial/scaling/linear parameters and adds `potential_tag = SUS2-SH`. For SH models, `alpha_index_basic` stores `{k, l, m}`. Internally this is converted to `mu = k * (l_max + 1) + l`, so the radial channel is still compatible with the SUS2 coefficient layout.

```text
B[k,l,m] = sum_j R_{k,l}(r_ij) Y_lm_real(rhat_ij)
```

The product graph stores real-basis Clebsch-Gordan contractions as `sh_products = {left, right, target, coeff}`. GPUMD must read these stored coefficients directly; they include the real tesseral basis phase convention used by training.

## l/k/body enumeration

Use `--body-l-max=body2,body3,body4,body5` for body-specific angular truncation through 5body, and `--body-l-max=body2,body3,body4,body5,body6` when `--body-order=6`. For example, `--l-max=3 --body-l-max=3,3,2,2,2` keeps 2-body and 3-body terms through `l=3`, while 4-body, 5body, and 6body terms use `l<=2`.

Factors are enumerated in canonical order with both `l1 >= l2 >= ...` and `k1 >= k2 >= ...`. Only even parity scalar paths are kept. The graph is built as the minimal shared DAG closure of the selected scalar targets; unused `(k,l,m)` channels and unused tensor-product intermediates are not generated.

The 6body path uses five SH factors and the rank-diagonal standard tree:

```text
A = (q0 x q1) -> L
B = (q2 x q3) -> L
C = (B  x q4) -> L
S = (A  x C ) -> 0
```

This is an intentional truncation of the full 2+3 tree. It preserves the scalar construction in a standard CG tree while reducing the number of 6body scalar paths by enforcing the same intermediate rank `L` on the pair and triple branches.

## l3k3 scalar counts

With `k=3`, `l=3`, even parity, and all body orders through 5:

- `--body-l-max=3,3,3,3`: 2-body 3, 3-body 24, 4-body 80, 5-body 615, total 722 scalars.
- `--body-l-max=3,3,2,2`: the raw enumeration has 317 candidate scalars. After removing CG paths with zero intermediate tensors, the active model has 261 scalar basis functions.
- `--body-order=6 --body-l-max=3,3,2,2,2`: the rank-diagonal 6body extension generates 621 active scalar basis functions in total. This keeps the 261 active 2body-through-5body scalars and adds the rank-diagonal 6body scalars after the same zero-path pruning.

Zero intermediate tensors can arise when identical tensor factors are coupled into an exchange-antisymmetric rank. For example, `(q10_l2_k2 x q10_l2_k2) -> L=1` is zero because swapping the two identical `l=2` factors gives the CG phase `(-1)^(2+2-1) = -1`. The generator must combine CG records before allocating nodes, and must not keep scalars depending on these zero tensors.

## GPUMD export plan

The export should include the header, channel table, coupling table, scalar table, and trained parameters. The coupling table should be generated from the same stored `sh_products` graph so GPUMD does not reconstruct a different CG convention.

Current SH support is implemented for `l <= 4`. The l=4 implementation uses the same real tesseral convention as the existing l<=3 path:

- `m = 0`: real `Y_l0`.
- `m > 0`: `sqrt(2) * (-1)^m * Re(Y_lm)`.
- `m < 0`: `sqrt(2) * (-1)^|m| * Im(Y_l|m|)`.

## Current Cu-Zr test run

Server project:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex
```

Training path:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex/run_cuzr_l3k3_b5_l3322
```

Model initialization:

```bash
../bin/mlp-sus2 init-sh untrained_sus2sh_l3k3_b5_l3322.mtp \
  --species-count=2 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss
```

As of the inline-init update, `init-sh` writes only the SH topology and radial
metadata. It intentionally omits `species_coeffs` and `moment_coeffs`, so
`MLMTPR::Load()` treats the model as uninitialized and the ordinary training
path initializes/fits parameters. Training can also build the same topology
directly:

```bash
../bin/mlp-sus2 train train.cfg --init-sh \
  --species-count=2 \
  --l-max=3 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0 \
  --do-samp \
  --do-lin \
  --do-lin-rescale \
  --curr-pot-name=current.mtp \
  --trained-pot-name=p.mtp
```

The corrected generated model has 48 basic SH channels, 1416 product graph records, 525 total moment nodes, and 261 scalar basis functions. The previous bad graph with 317 scalar basis functions was archived as `untrained_sus2sh_l3k3_b5_l3322.bad-shgraph.mtp`; it kept zero intermediate tensors such as nodes 171/172/173 and produced structural zero descriptor columns.

The l=4 probe model was generated with:

```bash
../bin/mlp-sus2 init-sh untrained_sus2sh_l4k3_b5_l4422.mtp \
  --species-count=2 \
  --l-max=4 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=4,4,2,2 \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss
```

The generated l=4/body-l-max `4,4,2,2` model has 75 basic SH channels, 1470 product graph records, 558 total moment nodes, and 267 scalar basis functions.

Single-configuration `calc-efs`, descriptor generation on `smoke5.cfg`, and a 5-configuration one-step BFGS smoke train complete successfully. The fixed descriptor check found zero structural all-zero columns on `smoke5.cfg`.

The Cu-Zr training file used here has energy and force labels but no stress/virial labels, so the submitted queue-33 job explicitly uses `--stress-weight=0`.

Queue-33 full training script:

```text
bsub_sus2sh_l3k3_b5_l3322_33.lsf
```

The active run uses 160 MPI ranks across four 40-core nodes, uses Hydra with the LSF hostfile, includes `--do-samp`, and keeps the SUS2 `do-lin` flow:

```bash
mpiexec.hydra -machinefile $LSB_DJOB_HOSTFILE -np $NP \
  /work/phy-weigw/20260321_Test/SUS2-SH-work-codex/bin/mlp-sus2 train \
  untrained_sus2sh_l3k3_b5_l3322.mtp \
  /work/phy-weigw/hyx/cu-zr/sus2-sh/train.cfg \
  --energy-weight=1 \
  --force-weight=0.01 \
  --stress-weight=0 \
  --stdd-weight=0.0 \
  --std-weight=0.0 \
  --do-samp \
  --init-params=same \
  --do-lin \
  --do-lin-rescale \
  --do-lin-steps=5000 \
  --do-lin-freq=50 \
  --max-iter=10000 \
  --curr-pot-name=current.mtp \
  --trained-pot-name=p.mtp
```

The first 40-rank test job was stopped and its logs were preserved under `killed_3657712/`. The bad-graph 160-rank job `3657721` was stopped and archived under `killed_3657721_bad_sh_graph/`. The corrected 261-scalar job is `3657797`.

## SH runtime optimization notes

The current safe optimizations keep the SH mathematical basis, CG coupling graph, scalar list, and parameter layout unchanged:

- The value-only forward path now evaluates only real SH values and skips SH coordinate derivatives.
- The evaluator avoids `std::pow` for `l <= 4` and uses explicit inverse-radius products.
- Small SH value/derivative work arrays are stack buffers (`lmax <= 4`) instead of per-call heap vectors.
- Per-neighbor radial contractions are cached by `mu=(k,l)` and reused across all `m` channels. This removes repeated `R`-dimension dot products in forward, force derivative, site-energy derivative, and direct BFGS coefficient-gradient paths.
- The linear-equation assembly skips force or stress component construction when the corresponding training weight is zero. This is important for the current Cu-Zr dataset because it has no stress labels and is trained with `--stress-weight=0`.
- Runtime SH evaluation now uses direct real spherical harmonic polynomials for values and coordinate derivatives. The older complex evaluator remains only as an internal reference path; ordinary forward, force, and BFGS gradient paths use the real `double` evaluator.
- Direct real SH normalization constants are initialized once and reused, instead of recomputing `sqrt(...)` factors for every neighbor.
- Force-weighted BFGS coefficient-gradient accumulation now caches each neighbor's real SH values/derivatives, radial basis values/derivatives, and `mu=(k,l)` radial contractions. This removes the previous duplicate `EvalRealSH`/`RB_Calc` pass inside `AccumulateSHCombinationGrad` when force residuals are active.
- The force/stress linear block buffers use `resize` instead of zero-filling `assign`, because the block rows are fully overwritten before BLAS assembly.

Validation against the pre-optimization universal binary:

- `smoke5.cfg` `calc-efs`: exact file match.
- `smoke5.cfg` one-step training: identical `current.mtp`.
- First 1000 Cu-Zr training structures `calc-efs`: numeric token max absolute difference `0.0`; text diff only showed `+0.00000/-0.00000` stress formatting.
- First 1000 Cu-Zr training structures one-step training: same metrics; `current.mtp` max numeric token difference `1.735e-18`.
- First 1000 Cu-Zr training structures timing on one rank: `calc-efs` improved from about `9.05 s` to `7.01 s`; one-step training improved from about `67.07 s` to `57.38 s`.
- Stress-gating validation with `--stress-weight=0`: `smoke5.cfg` one-step linear path produced identical `current.mtp`. A 300-structure l=4 linear solve reduced elapsed time from about `136.3 s` to `126.3 s`.
- Stress-active validation with synthetic `PlusStress` and `--stress-weight=0.01`: `current.mtp` matched the pre-gating binary exactly.
- l=4 regression: `calc-efs` and one-step training completed on `smoke5.cfg`; finite-difference force validation on six atom/direction components gave max absolute error about `5.0e-7 eV/A`.
- Direct-real SH formula check: independent pointwise comparison against the previous complex evaluator found l=4 value differences around `4.4e-16` and derivative differences around `5.3e-15`.
- Direct-real runtime regression against the complex-evaluator backup: l3/l4 `calc-efs` matched exactly; deterministic one-step BFGS from complete models matched exactly for l3 and to about `1.7e-11` max numeric token difference for l4.
- Direct-real final finite-difference l=4 check: six force components retained max absolute error about `5.0e-7 eV/A`.
- Direct-real `calc-efs` timing on the first 1000 Cu-Zr structures, one rank, four alternating old/new samples: complex-evaluator mean `6.82 s`, direct-real mean `6.37 s` (about `1.07x`). The login-node timing is noisy, so this should be treated as indicative rather than a hard benchmark.
- Direct-real one-step BFGS timing on 300 structures from a complete model did not improve in the short run (`26.0 s` old vs `26.8 s` new), while the trained parameter files matched to `2.6e-18`. This suggests the current training bottleneck is dominated by coefficient-gradient assembly and linear algebra around the SH values, not by complex-vs-real SH evaluation alone.
- SH neighbor-cache validation against `bin/mlp-sus2.before-sh-neighbor-cache-20260513`: l3 `smoke5.cfg` with fixed `ini.mtp --skip-preinit` matched exactly for `calc-efs`, `current.mtp`, and `p.mtp`.
- SH neighbor-cache l4 validation with the l4/body-l-max `4,4,2,2` model matched exactly for `calc-efs`, `current.mtp`, and `p.mtp`.
- SH neighbor-cache 300-structure BFGS timing from fixed `ini.mtp --skip-preinit`: old runs `26.23 s`, `24.90 s`; new runs `21.19 s`, `20.70 s`; mean speedup about `1.22x`. The resulting `current.mtp` numeric max difference was `1.735e-18`.

Rejected or not-yet-retained optimization attempts:

- Lowering the force block BLAS threshold from `n >= 1000` to `n >= 128` was not retained. On a 300-structure `--do-lin` test it changed the linear solution and training metrics, likely because the normal equation is ill-conditioned enough that different BLAS summation order changes the solve. Keep this as an opt-in experiment only after a dedicated normal-matrix/RHS audit.
- Replacing selected SH gradient inner-loop `mu_to_*` lookups with `basic_*_cache_` was not retained as a standalone change. It is mathematically expected to be equivalent, but it did not provide a useful validated speed result and must be reintroduced only with fixed-start BFGS regression tests.

Larger but more invasive optimization candidates are intentionally deferred:

- Stream force/stress rows directly into the linear normal equations instead of materializing full `basis_ders`, `forces_cmpnts`, and `stress_cmpnts`.
- Fuse prediction and BFGS residual-gradient accumulation so a neighborhood's SH/radial state is not rebuilt once for E/F prediction and again for coefficient gradients.
- Add a final scalar-rooted graph closure/reindexing pass if future scalar generation leaves unreachable product nodes.
