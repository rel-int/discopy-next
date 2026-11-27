`drawing.Drawing`, `python.finset.Cospan` and `interaction.Diagram`
declare `abc.DaggerCategory`, the reflection of a drawing being
involutive, with its contravariance and monoidality declared failing on
the nose, and the dagger laws of `Int(compact.Diagram)` declared failing,
feedback having no dagger; `python.additive.Function.dagger`, defined on
swaps alone, is `inverse`; and a functor maps a daggered box as the dagger of its image
when its codomain is a `DaggerCategory` that `has_dagger`, which integer
diagrams over a category with no dagger do not, not whenever it has a
`dagger` ([next#631](https://github.com/rel-int/discopy-next/pull/631)).
