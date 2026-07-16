# sip_qp
[![pip](https://github.com/joaospinto/sip_qp/actions/workflows/pip.yml/badge.svg)](https://github.com/joaospinto/sip_qp/actions/workflows/pip.yml)
[![wheels](https://github.com/joaospinto/sip_qp/actions/workflows/wheels.yml/badge.svg)](https://github.com/joaospinto/sip_qp/actions/workflows/wheels.yml)

`sip_qp` is a quadratic-programming front-end to
[SIP](https://github.com/joaospinto/sip). It equilibrates affine QPs, applies
QP-oriented solver defaults, and uses
[SIP-QDLDL](https://github.com/joaospinto/sip_qdldl) for sparse Newton-KKT
solves.

Both dynamically allocated and caller-provided workspaces are supported. The
solve path performs no dynamic memory allocation when `Workspace::mem_assign`
is used; this property is covered by an allocation-guarded test.

The C++ API represents

```text
minimize  objective_constant + q' x + 0.5 x' P x
subject to A x + equality_offsets = 0
           G x + inequality_offsets <= 0
           lower_bounds <= x <= upper_bounds.
```

`P` is supplied by its upper triangle. Constraint Jacobians are supplied
transposed, as expected by SIP-QDLDL. Bound arrays are always present; infinite
entries represent missing sides.

`Input` requires an inverse KKT permutation and its symbolic factor capacity.
Callers may provide both directly. `analyze_kkt_ordering` validates a
caller-defined inverse permutation and computes its factor capacity. Ordering
policy remains entirely outside this library.

The `sip_qp_python` package exposes the same implementation through `Solver`
and `solve_qp`. Python callers must pass `kkt_inverse_permutation`. Variable
bounds remain native SIP bounds and are reported separately through
`Result.z_box`.
