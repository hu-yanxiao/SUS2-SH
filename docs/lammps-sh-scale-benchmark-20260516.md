# SUS2-SH LAMMPS Scale Benchmark 2026-05-16

This benchmark measures the SUS2-SH LAMMPS `_lmp` radial-table path across
system sizes, comparing one A100 GPU against 80 CPU MPI ranks.

## Setup

- Model: Cu-Zr SUS2-SH 3322 `_lmp`
- Data source: 128-atom Cu-Zr base cell extracted from the existing 10,240-atom
  benchmark data and replicated to 1,024, 10,240, 102,400, and 1,024,000 atoms
- GPU binary:
  `/work/phy-weigw/20260321_Test/lammps-sus2-sh-work-codex/bin/lmp.sus2_sh_kk_sm80`
- CPU binary:
  `/work/phy-weigw/20260321_Test/lammps-sus2-sh-work-codex/bin/lmp.sus2_sh_cpu_avx2`
- GPU run: 1 MPI rank, 1 A100, Kokkos CUDA, `neigh half newton on comm device`
- CPU run: 80 MPI ranks, 2 CPU nodes with 40 ranks per node
- Dynamics: `run 1000`, `fix nve`, no dump or final data output
- Benchmark directory:
  `/work/phy-weigw/20260321_Test/lammps-sus2-sh-work-codex/tests/scale_lmp_20260516`

The final step-1000 thermodynamic lines matched between GPU and CPU to printed
precision for all four sizes.

## Results

Speedup is `CPU80 loop time / A100 loop time`, so values above 1 mean the A100
is faster than 80 CPU MPI ranks.

| Atoms | A100 loop s | CPU80 loop s | A100 Matom-step/s | CPU80 Matom-step/s | Loop speedup | A100 pair s | CPU80 pair avg s | Pair speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 7.31949 | 1.60277 | 0.1399 | 0.6389 | 0.2190 | 3.7691 | 1.0718 | 0.2844 |
| 10,240 | 10.0983 | 11.9123 | 1.0140 | 0.8596 | 1.1796 | 6.4123 | 10.852 | 1.6924 |
| 102,400 | 99.7581 | 114.951 | 1.0265 | 0.8908 | 1.1523 | 96.039 | 108.93 | 1.1342 |
| 1,024,000 | 994.556 | 1130.38 | 1.0296 | 0.9059 | 1.1366 | 990.49 | 1088.9 | 1.0994 |

## Interpretation

At 1,024 atoms, the A100 path is dominated by fixed Kokkos/GPU communication
overhead, so the 80-rank CPU run is much faster. Starting at 10,240 atoms, the
A100 becomes faster in total loop time. From 10,240 to 1,024,000 atoms, the
total-loop speedup is stable at about `1.14-1.18x`, consistent with one A100
being roughly comparable to, and slightly faster than, 80 CPU MPI ranks for the
current SUS2-SH LAMMPS `_lmp` path.

The pair-only speedup decreases with system size from `1.69x` at 10,240 atoms
to `1.10x` at 1,024,000 atoms. This suggests the current A100 implementation is
not yet exposing a large SH-specific matrix/tensor parallel advantage in LAMMPS;
for large systems it is mainly slightly ahead of the 80-core CPU path rather
than substantially faster.
