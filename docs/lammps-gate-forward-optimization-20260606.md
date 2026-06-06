LAMMPS two-layer gate forward optimization log, 2026-06-06

Goal
----
Recover the previous ML-SUS2 LAMMPS forward speed for the current tanh-gated
two-layer SH model while keeping the exact model definition:

  R_ij,mu(r) is multiplied by v_Zj * (1 + A * tanh(a_Zj,mu * f_j)).

Constraints
-----------
- Keep the main one-layer/full SH path mathematically unchanged.
- Keep the second-layer/main SH pass dense and complete for every lk model.
- Do not use sparse gate graph pruning unless it is rebuilt as a compact dense
  program and benchmarked.
- Keep _lmp table mode mathematically identical to direct radial evaluation.
- Benchmark old and new binaries in the same LSF job on the same 33-queue CPU
  node, and record hostname and CPU model.

Observed baseline
-----------------
The first pruning-oriented implementation was correct, but slower:

- old canonical LAMMPS: 6.7916 s for 200 steps
- new with tanh cache disabled: 8.73674 s
- new with tanh/multiplier cache enabled: 7.89491 s

This shows the per-atom/per-mu gate cache is useful, but sparse graph pruning
adds enough indirect access overhead to lose to the original dense loops.

Current plan
------------
1. Return the inner gate SH pass to dense moment/product/basic loops.
2. Keep the per-atom/per-mu gate multiplier and derivative cache.
3. Cache edge table interpolation metadata: table index, radial bin, and
   interpolation fraction, then reuse it in the main additive pass.
4. Add OpenMP SIMD reductions to dense force/backprop dot-product loops.
5. Verify correctness with:
   - dev_test/lammps_two_layer_gate_additive_mlp_check.sh
   - dev_test/lammps_two_layer_gate_lmp_smoke.sh
6. Benchmark old/new/new-cache-disabled in one LSF job.

Implementation notes
--------------------
Local edit round 1:

- Removed the sparse gate dependency closure from the LAMMPS interface.
- Restored the inner gate pass to dense `forward_sh_products()` and
  `backprop_sh_products()` over the full SH product graph.
- Restored `accumulate_sh_basic_edge()` to a dense full-basic loop.
- Kept the per-atom/per-mu gate multiplier and derivative cache.
- Added edge table metadata caches:
  - table index
  - radial table bin
  - interpolation fraction
- Reused the edge table metadata in the second additive pass.
- Added OpenMP SIMD reductions to the dense gate-derivative and main force
  dot-product loops.

Verification notes
------------------
Server no-IPO build:

  module purge
  module load gcc/11.2.0 compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0
  cd /work/phy-weigw/apps/lammps-10Dec2025/src
  make -j8 ml_sus2_avx2_noipo

Result:

- Built `/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo`.
- Binary size output: text=10115461, data=215568, bss=13608.

Correctness checks with the no-IPO binary:

- `dev_test/lammps_two_layer_gate_additive_mlp_check.sh`
  - `mlp_calc_efs vs lammps_1: max_abs=8.789600e-05`
  - `mlp_calc_efs vs lammps_2: max_abs=8.789600e-05`
  - `lammps_1 vs lammps_2: max_abs=0.000000e+00`
  - result: ok
- `dev_test/lammps_two_layer_gate_lmp_smoke.sh`
  - direct/table differences all `0.000000e+00`
  - result: ok

No-IPO performance check
------------------------
Benchmark directory:

  /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/md/codex_lmp_dense_cache_table_noipo_bench_20260606_181550

LSF job:

  3759747

Node:

- hostname: `b08u21a`
- CPU model: `Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz`
- allocation: `40*b08u21a`

Loop times for 200 steps, 3051 atoms:

- old canonical: `5.79037 s`
- new no-IPO, cache disabled: `7.09084 s`
- new no-IPO, cache enabled: `6.01505 s`

Interpretation:

- The dense-loop rollback plus table metadata cache recovers most of the
  slowdown from the pruning implementation.
- The gate multiplier/derivative cache reduces the new no-IPO loop time from
  `7.09084 s` to `6.01505 s`, a runtime reduction of about 15.2%.
- The verified no-IPO new binary is still about 3.9% slower than the old
  canonical binary in this benchmark. This is not an install-ready speed
  result yet because the formal AVX2/IPO build has not been completed.

Formal AVX2/IPO build and benchmark
-----------------------------------
The formal AVX2/IPO build was submitted as LSF job `3759749` on `b08u21b`.
The build completed successfully and produced:

  /work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2

Binary hash:

  fa63dc8ff52cc66347a6cf1fe613c095290c71511c3fe7633354e1776babf993

Correctness checks with the formal AVX2/IPO binary:

- `dev_test/lammps_two_layer_gate_additive_mlp_check.sh`
  - `mlp_calc_efs vs lammps_1: max_abs=8.789600e-05`
  - `mlp_calc_efs vs lammps_2: max_abs=8.789600e-05`
  - `lammps_1 vs lammps_2: max_abs=0.000000e+00`
  - result: ok
- `dev_test/lammps_two_layer_gate_lmp_smoke.sh`
  - direct/table differences all `0.000000e+00`
  - result: ok

Formal benchmark on Xeon Gold 6133:

- directory:
  `/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/md/codex_lmp_dense_cache_table_ipo_bench_20260606_183055`
- job: `3759763`
- node: `b07u11a`
- CPU: `Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz`
- old canonical: `6.96603 s`
- new AVX2/IPO, cache disabled: `7.54112 s`
- new AVX2/IPO, cache enabled: `6.73798 s`
- interpretation: new default is about 3.3% faster than old canonical on this
  CPU type.

Formal benchmark on Xeon Gold 6138:

- directory:
  `/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/md/codex_lmp_dense_cache_table_ipo_bench_6138_20260606_183240`
- job: `3759773`
- node: `b08u21a`
- CPU: `Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz`
- old canonical: `5.73750 s`
- new AVX2/IPO, cache disabled: `6.51964 s`
- new AVX2/IPO, cache enabled: `5.78139 s`
- interpretation: new default is about 0.8% slower than old canonical on this
  CPU type. This is close enough that an alternated repeat benchmark is needed
  before deciding whether to install this binary as the canonical CPU LAMMPS.

Formal repeat benchmark on Xeon Gold 6138:

- directory:
  `/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/md/codex_lmp_dense_cache_table_ipo_repeat_6138_20260606_183450`
- job: `3759774`
- node: `b08u21a`
- CPU: `Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz`
- order: old, new, new, old
- old canonical:
  - `old_1: 5.76757 s`
  - `old_2: 5.97066 s`
- new AVX2/IPO, cache enabled:
  - `new_1: 5.79002 s`
  - `new_2: 5.76176 s`
- interpretation: the first old/new pair is within 0.4%; the second pair is
  dominated by a slower `old_2`. The mean of the two new runs is about 1.9%
  faster than the mean of the two old runs. The result supports installing the
  new binary as a speed-neutral or slightly faster replacement on the 6138 CPU
  type.

Installation
------------
Installed the formal AVX2/IPO binary to the canonical CPU LAMMPS paths:

- `/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi`
- `/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi_avx2`

Installed hash:

  fa63dc8ff52cc66347a6cf1fe613c095290c71511c3fe7633354e1776babf993

Previous canonical binary backup:

  /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.bak_20260606_183600_dense_cache_table

Backup hash:

  79d24851095c12606e1b9b2be9ff8bb33d547e30a3dffcfd53424032fcd5a58f

Post-install correctness checks using the canonical installed path:

- `dev_test/lammps_two_layer_gate_additive_mlp_check.sh`: ok
  - `mlp_calc_efs vs lammps_1: max_abs=8.789600e-05`
  - `mlp_calc_efs vs lammps_2: max_abs=8.789600e-05`
  - `lammps_1 vs lammps_2: max_abs=0.000000e+00`
- `dev_test/lammps_two_layer_gate_lmp_smoke.sh`: ok
  - direct/table differences all `0.000000e+00`
