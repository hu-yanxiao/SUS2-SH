# SUS2-SH LAMMPS Optimization Log

This note records tested LAMMPS GPU and CPU optimization attempts for the
SUS2-SH interface. GPU timings use the Cu-Zr 10k atom 3322 `_lmp` test, one
A100 (`sm_80`), `run 2000`, double precision, and direct radial table mode
unless noted otherwise. CPU timings use the same 10,240-atom Cu-Zr 3322
`_lmp` model, `run 2000`, double precision, 40 MPI ranks, Intel AVX2 build.

## Accepted GPU Changes

| Label | Main change | Loop time | Pair time | Status |
| --- | --- | ---: | ---: | --- |
| baseline before SH indexing | Previous formal Kokkos path | 29.0039 s | 28.371 s | Reference |
| shidx | Precomputed flattened SH indices for basic and grouped force paths | 24.906-24.9695 s | 24.276-24.339 s | Accepted |
| forcebase | Neighbor-level radial table base indices, edge-level `1/dist`, and skipped unused non-env `raw_contrib` | 24.3183-24.426 s | 23.687-23.793 s | Accepted |
| forcetmpl | Compile-time force specialization for SH/non-SH and env-gate on/off paths | 22.4287-22.4583 s | 21.797-21.825 s | Accepted |
| nozero | Remove full zero-fill of local SH value/derivative arrays; every used `(l,m)` component is written explicitly before use | 20.5828-20.6481 s | 19.932-19.986 s | Accepted |

The final thermo line for accepted candidates matched the `shidx` reference to
printed precision:

```text
2000 10240 -62831.532 265.67795 -62565.854 200.73973 218.94537 183538.51 66.096749 52.901597 52.490212 0 0 0
```

## Accepted CPU Changes

| Label | Main change | Old loop | New loop | Old pair avg | New pair avg | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu-basic-shpath | Cache flattened SH component per basic alpha, split SH env-gate/no-env branches, and skip moment-tensor coordinate/distance power tables on the SUS2-SH path | 41.8932 s | 40.5244 s | 39.286 s | 38.394 s | Accepted |
| cpu-basic-shpath reverse-order check | Same binaries, but run new first and old second to check ordering effects | 41.549 s | 40.5227 s | 39.113 s | 38.394 s | Accepted |

The CPU final step-2000 thermodynamic line matched between the old and new
AVX2 binaries to printed precision:

```text
2000 10240 -62829.148 264.53068 -62564.618 199.87288 -62.536088 183165.01 65.981795 53.037901 52.339801 0 0 0
```

CPU build note: the formal AVX2 binary is built with Intel `-ipo`; the clean
build completed successfully but spent most of its 816 s runtime in the final
single-process IPO link. For future CPU micro-optimizations, keep this formal
build for final validation, but use a non-IPO or incremental experimental build
only for early screening.

## Rejected GPU Changes

| Label | Main change | Reason rejected |
| --- | --- | --- |
| randomaccess | Kokkos `RandomAccess` memory traits on read-only views | Slower than `shidx`. |
| shscratch | Extra shared-memory SH handling around basic accumulation | Slower than simpler scratch accumulation. |
| layerteam | Team-per-atom graph-layer product/reverse path | Math matched, but slower from team/block overhead. |
| layerflat | Flattened atom-by-layer product/reverse kernels | Math matched, but much slower from kernel launches and global traffic. |
| alphatmpl | Compile-time basic-alpha SH/env-gate specialization without scratch layout changes | Math matched, but slower than `forcetmpl` on the 10k test. |
| shpow | Precompute `r^{-0..4}`/derivative powers inside SH evaluation | Math matched, but nozero-only was slightly faster in direct A/B testing. |
| mucontract | For SH/no-env force, pre-sum `coeff * Y_lm` and `coeff * dY_lm` within each `mu_group` | Thermo matched to printed precision and was only about 0.3% faster than nozero; rejected for now because it changes floating-operation grouping and produced compiler unreachable-loop warnings. |

## Current Direction

Further work should target generic, model-independent reductions in force/basic
inner-loop cost and memory traffic. Avoid radial-basis-specific shortcuts.
The nozero result shows that the SH path is sensitive to local-array writes and
register pressure. The next high-confidence candidates should focus on the
basic-alpha kernel, grouped force contractions, and table/memory layout. The
force kernel now has compile-time specialization for the common
SUS2-SH/no-env-gate path, so repeating that strategy where it removes real
memory traffic is more promising than graph-layer rewrites. On CPU, the accepted
change shows that the SH path should avoid moment-tensor-only work entirely;
larger gains likely require product-path or memory-layout changes rather than
more small branch cleanup.
