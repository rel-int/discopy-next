A category draws left to right when its `Diagram` class sets
`draw_left_to_right = True`, and a box whose args are drawn in
another direction than its own category is drawn as one box with its args
across it, e.g. the "Very big car" figure of higher-order DisCoCat: the
terms top to bottom, their bodies left to right.
`drawing.backend.Transposed` draws through the reflection along the
anti-diagonal, keeping the layout as it is ([next#728](https://github.com/rel-int/discopy-next/pull/728)).
