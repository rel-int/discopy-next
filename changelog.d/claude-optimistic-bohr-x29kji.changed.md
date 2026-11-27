A functor from a symmetric category asks the level of its codomain rather
than its attributes: a permutation goes to one native box in a free
symmetric category, to the codomain's own `permutation` of the images of
the wires in any other `SymmetricCategory`, and swap by swap otherwise.
`stream.Stream` and `drawing.Drawing`, which had a swap without saying so,
are `SymmetricCategory`, and `Drawing.swap` draws one. The matrix then
checks `scalar_sliding` and `braid_naturality` on streams, up to the order
of their memory as `bifunctoriality` is ([next#514](https://github.com/rel-int/discopy-next/pull/514)).
