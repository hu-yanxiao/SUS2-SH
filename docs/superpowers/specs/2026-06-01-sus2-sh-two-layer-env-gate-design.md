# SUS2-SH Two-Layer Neighbor Scalar Gate Design

## Goal

Implement a non-recursive two-layer SUS2-SH model where the outer atom type factor in the final SH moment is modulated by a scalar environment descriptor of the outer atom:

```text
B_i^(1)(a) = sum_{b in N(a)} lambda_a lambda_b (1 + f_b) R_i(r_ab) Y_i(rhat_ab)
f_b = sum_q w_q Phi_q^(0)(b)
```

Here `i = (k,l,m)`, `Phi_q^(0)(b)` is built from an unmodulated first-layer SH scalar graph, and `w_q` are the only new trainable parameters.

## Mathematical Form

The first-layer basic moments are the existing unmodulated SUS2-SH moments:

```text
M_i^(0)(b) = sum_{c in N(b)} lambda_b lambda_c R_i(r_bc) Y_i(rhat_bc)
```

The gate scalars are a selected scalar subset of the existing SH product graph:

```text
Phi_q^(0)(b) = G_q({M_i^(0)(b)})
q in Q_gate, body_order(q) <= gate_body_order_max
```

The default gate body cutoff is `gate_body_order_max = 3`, while the final model can keep a larger scalar body order such as `6`.

The second-layer final moments use the neighbor-local gate value:

```text
M_i^(1)(a) = sum_{b in N(a)} lambda_a lambda_b g_b R_i(r_ab) Y_i(rhat_ab)
g_b = 1 + f_b
f_b = sum_{q in Q_gate} w_q Phi_q^(0)(b)
```

The final site energy remains the ordinary SUS2-SH linear form over the final scalar basis:

```text
E_a = shift_{t_a} + c_{t_a} (1 + sum_p beta_p Phi_p^(1)(a))
```

The gate is not recursive. `Phi_q^(0)` is always built from unmodulated `M^(0)`, not from `M^(1)`.

One-body terms are excluded from the first-layer gate. The existing per-species shift and species linear terms remain only in the second-layer site-energy expression.

## Coefficient Layout

The existing nonlinear block is:

```text
[shift_coeffs][scal_coeffs][radial_coeffs]
```

For a gated model it becomes:

```text
[shift_coeffs][scal_coeffs][radial_coeffs][two_layer_gate_weights][linear coeff mirror]
```

The linear coefficients remain stored and solved in the existing `linear_coeffs` block and mirrored at the tail of `regression_coeffs` by `LinCoeff()`. Gate weights are nonlinear coefficients and are not part of the linear solve. They are initialized to zero so `g_b = 1` is an exact compatibility mode.

## Metadata

Gated models must write explicit metadata instead of inferring the gate subset from the final graph:

```text
two_layer_gate_enabled = true
two_layer_gate_body_order_max = 3
two_layer_gate_include_one_body = false
two_layer_gate_weight_count = <|Q_gate|>
two_layer_gate_scalar_indices = { ... indices into alpha_moment_mapping ... }
two_layer_gate_weights = { ... }
```

For gated models `sh_scalar_info` is mandatory because the body-order cutoff must be exact and reproducible. Plain SH models may continue to omit it.

## Efficient Training Runtime

The correct training path is configuration-level, not per-center-only:

1. Build or reuse the ordinary `r_c` neighborhoods for every atom.
2. First layer: compute unmodulated `M^(0)` and selected `Phi^(0)` for all atoms.
3. Compute `f_i` and `g_i = 1 + f_i` for every atom.
4. Second layer: compute final gated moments and scalar basis for every center using `g_neighbor`.
5. Accumulate energy, forces, stresses, linear components, and nonlinear gradients at configuration scope.
6. Reverse mode must send `dL/df_b` from final edges into the gate graph for atom `b`, then through atom `b`'s first-shell neighbors.

This is required because an energy term centered at `a` can depend on an atom `c` that is not a direct neighbor of `a` when `c` is a neighbor of `b`. The effective mathematical dependency radius is `2*r_c`, but the implementation should keep normal `r_c` neighbor lists and pass scalar/adjoint information between layers.

## LAMMPS And GPUMD Direction

LAMMPS should use scalar communication rather than a physical `2*r_c` halo:

1. Owned atoms compute `Phi^(0)` and `f`.
2. Forward-communicate `f` to ghost copies.
3. Main pair pass uses neighbor `f_j`.
4. Reverse-communicate accumulated `dE/df_j`.
5. Owned atoms backpropagate the gate graph and add indirect force/virial terms.

GPUMD can keep the same global atom-array shape and add device kernels in this order:

```text
gate basic -> gate forward -> main gated SH -> main reverse with gate adjoints -> gate reverse force
```

Neither interface should build a real `2*r_c` neighbor cache for the main design.

## Validation Contract

Required correctness checks:

1. Plain SH model output remains unchanged.
2. Gated model with all `w_q = 0` matches the same topology without gate weights to numerical precision.
3. Model save/load preserves gate metadata and gate weights.
4. A finite-difference force test includes a two-hop atom `c` that is outside atom `a`'s direct shell but inside atom `b`'s shell.
5. A finite-difference coefficient-gradient test includes at least one gate weight.
6. Remote 500-configuration, 100-step training compares old one-shot SH and the two-layer gate for accuracy and wall time.
