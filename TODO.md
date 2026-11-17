> Build slice 2 of rel-int/discopy-next#134, the plan USER (toumix) ruled live 10-19 for `discopy.monad`: products, powers and operations of Eilenberg–Moore algebras (an operation t ∈ M(n) acts on every algebra), and the writer monad G × - whose algebras are `group.Action`s, in `discopy/monad/algebra.py`.

- [x] The mathematics in the module docstring: products, powers, operations, the writer monad and its algebras as G-actions
- [x] `Algebra.product` and `Algebra.power`, tested on the unit and associativity laws
- [x] `Operation[M]`: t ∈ M(n) as a map Aⁿ → A on every algebra, tested commuting with homomorphisms
- [x] `Writer(G)`, with `from_action` and `to_action` round-tripping with `group.Action` on a small group
- [x] CHANGELOG `Added` bullet, next-checks green
