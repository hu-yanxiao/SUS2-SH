# SUS2-SH

Standalone C++ prototype for a spherical-harmonic SUS2 developer branch.

The first target is to make the training-side math path concrete:

- basic moments are `B[k,l,m] = sum_j R_{k,l}(r_ij) Y_lm(rhat_ij)`;
- scalar features are O(3)-even Clebsch-Gordan contractions;
- `body_order = 5` means up to four SH factors around one center atom;
- the standard four-factor tree is `((q1 x q2) -> L, (q3 x q4) -> L) -> 0`;
- leaf deduplication is canonical by descending `(l,k)`.

This is not yet the production `mlp-sus2` integration. The production training
logic should stay the same as SUS2-MLIP: SUS2-SH replaces the angular basis,
scalar generation, and chain-rule derivative path, not the outer C++ trainer.
The standalone command here is a smoke harness used to validate topology, file
format, and data flow before porting the kernel into the existing C++ training
framework.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Inspect l3k3, 5-body cutoff

```bash
build/sus2-sh inspect --l-max 3 --k-max 3 --body-order 5
```

Expected prototype topology:

```text
scalar_count_without_bias = 1409
scalar_count_with_bias = 1410
```

The four-factor count uses all standard-tree intermediate `L` channels after
canonical leaf ordering. A later symmetry-pruned basis can reduce this count,
but the first training path keeps the full standard-tree contraction plan.

## Energy-Only Smoke Harness

```bash
build/sus2-sh train \
  --train-cfg train.cfg \
  --model-out sus2-sh-l3k3-b5.model.json \
  --l-max 3 \
  --k-max 3 \
  --body-order 5 \
  --cutoff 7.5 \
  --radial-basis-size 10 \
  --radial-basis-type RBChebyshev_sss \
  --max-configs 256
```

Current limitations:

- linear energy-only normal-equation smoke trainer;
- fixed initial radial scaling slots per `k`;
- force chain-rule path is present in `check-grad`, but force/stress loss is not
  wired into the SUS2 optimizer yet;
- no GPUMD runtime kernel yet.

## Chain-Rule Check

```bash
build/sus2-sh check-grad \
  --train-cfg train.cfg \
  --l-max 3 \
  --k-max 3 \
  --body-order 5 \
  --cutoff 7.5 \
  --radial-basis-size 10 \
  --radial-basis-type RBChebyshev_sss
```
