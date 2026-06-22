# SUS2-SH Spatial ACE Plan

## Goal

Implement spatial ACE support on branch `codex/spatial-ace` from current main.
The branch should contain:

- a strict-equivalent backend that reproduces the existing SUS2-SH coupling
  graph and output exactly within numerical tolerance;
- a separate direct spatial ACE variant for experiments where the basis is
  intentionally changed;
- independent server build and benchmark paths;
- implementation and optimization history for later continuation.

## Implementation Slices

1. Add a strict CG-map verifier.
   - Add `check-sh-spatial-ace-cg-map-dev`.
   - Verify the spatial/SO2-friendly map reproduces the existing real-basis CG
     convention through `lmax=4`.
   - Keep this independent of training and model IO.

2. Add a strict backend switch.
   - Introduce an opt-in runtime switch, initially environment-controlled, so
     main behavior remains unchanged by default.
   - Implement forward product evaluation for the existing `sh_products_`
     graph.
   - Add reverse propagation for product derivatives.

3. Validate plain single-layer SH.
   - Generate `l2k2`, `l3k3`, and `l4k4` models from the same init options.
   - Compare strict backend output to default main semantics on the same CFGs.
   - Run force/stress finite-difference checks.

4. Validate two-layer gate mode.
   - Reuse existing gate checks and add backend equivalence coverage.
   - Run parameter-gradient, force, and stress finite-difference checks.

5. Add direct spatial ACE as an explicit non-equivalent model variant.
   - Use separate options and file markers so direct models cannot be confused
     with strict-equivalent models.
   - Add focused tests documenting the expected low-order behavior and the
     non-equivalence boundary.

6. Remote build and efficiency comparison.
   - Create independent server source/build path for this branch.
   - Benchmark strict backend against the current main reference on:

```text
/work/phy-weigw/hyx/xxx-b/spatial_ace/
```

## Optimization Notes

- Do not use approximation-based speedups for gate mode by default.
- Prefer exact layout, scheduling, cache reuse, recomputation, and memory-flow
  improvements.
- Compare at equivalent call counts and model settings before drawing speed
  conclusions.
