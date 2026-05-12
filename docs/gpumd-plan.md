# GPUMD Plan For SUS2-SH

GPUMD implementation is intentionally deferred until the C++ training path is
stable.

Planned model sections:

- `format = SUS2-SH`
- `l_max`, `k_max`, `body_order`, `parity = even_O3`
- radial basis metadata and per-`k` scaling parameters
- block SH basic channels `(k,l)` with implicit contiguous `m=-l..l`
- tensor product plan records:
  - source block ids
  - output `L`
  - coefficient offset/count
  - tree stage id
- Clebsch-Gordan coefficient table
- scalar mapping and linear coefficients

Planned GPU execution:

1. Build `B[k,l,m]` for every center atom.
2. Run tensor-product forward by grouped `(l1,l2,L)` contractions.
3. Seed scalar adjoints from linear coefficients.
4. Run reverse tensor-product contractions to obtain `dE/dB[k,l,m]`.
5. Run edge force kernel using `R`, `R'`, `Y_lm`, and `dY_lm/dr`.

Derivative formula target:

```text
d/dx_a [R_{k,l}(r) Y_lm(rhat)]
= R'_{k,l}(r) n_a Y_lm(rhat)
  + R_{k,l}(r) dY_lm(rhat)/dx_a
```

For `l<=3`, generate explicit solid-harmonic formulas for `Y` and `dY`.
Do not store a full `edge x channel x xyz` Jacobian on GPU. Recompute the
small edge-local SH/radial derivative table and contract it with `dE/dB`.

