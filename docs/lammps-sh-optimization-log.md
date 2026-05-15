# SUS2-SH LAMMPS Optimization Log

This note records tested LAMMPS Kokkos/GPU optimization attempts for the SUS2-SH
interface. Timings below use the Cu-Zr 10k atom 3322 `_lmp` test, one A100
(`sm_80`), `run 2000`, double precision, and direct radial table mode unless
noted otherwise.

## Accepted Changes

| Label | Main change | Loop time | Pair time | Status |
| --- | --- | ---: | ---: | --- |
| baseline before SH indexing | Previous formal Kokkos path | 29.0039 s | 28.371 s | Reference |
| shidx | Precomputed flattened SH indices for basic and grouped force paths | 24.906-24.9695 s | 24.276-24.339 s | Accepted |
| forcebase | Neighbor-level radial table base indices, edge-level `1/dist`, and skipped unused non-env `raw_contrib` | 24.3183-24.426 s | 23.687-23.793 s | Accepted candidate |

The final thermo line for accepted candidates matched the `shidx` reference to
printed precision:

```text
2000 10240 -62831.532 265.67795 -62565.854 200.73973 218.94537 183538.51 66.096749 52.901597 52.490212 0 0 0
```

## Rejected Changes

| Label | Main change | Reason rejected |
| --- | --- | --- |
| randomaccess | Kokkos `RandomAccess` memory traits on read-only views | Slower than `shidx`. |
| shscratch | Extra shared-memory SH handling around basic accumulation | Slower than simpler scratch accumulation. |
| layerteam | Team-per-atom graph-layer product/reverse path | Math matched, but slower from team/block overhead. |
| layerflat | Flattened atom-by-layer product/reverse kernels | Math matched, but much slower from kernel launches and global traffic. |

## Current Direction

Further work should target generic, model-independent reductions in force/basic
inner-loop cost and memory traffic. Avoid radial-basis-specific shortcuts. The
next high-confidence candidate is compile-time specialization for the common
SUS2-SH/no-env-gate path so that the force kernel does not carry runtime SH and
environment-gate branches inside the innermost loops.
