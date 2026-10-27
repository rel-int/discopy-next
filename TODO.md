> what do you mean discopy drops scalars? that shouldn’t be the case we definitely wanna keep all scalars

Design, settled with USER in session: the scalar is an attribute of `zx.Hypergraph`, exact as
`√2^k · e(φ) · Π Σ c e(ψ)` with a complex factor only for numbers that come in as such.

- [WIP] @01PDixeG-2026-10-27 08:10 `zx.Number`, the exact scalar: product, conjugate, value, tree
- [ ] `zx.Hypergraph.scalar`: multiplied by `then` and `tensor`, conjugated by `dagger`, kept by every
  constructor, read by equality, hash, `repr`, tree; `Scalar` boxes absorbed, `to_diagram` gives one back
- [ ] every rule of `zx.Rewriter` an equation on the nose, its right side carrying the factor
- [ ] `from_pyzx` and `to_pyzx` keep the scalar of a PyZX graph
- [ ] tests exact rather than proportional, CHANGELOG, PR description
