The TikZ backend draws no black frame around a drawing: `TikZ.draw_polygon`
writes the `edgecolor` it is given, e.g. the white edge of the background.
A spider is an empty node padded by `nodesize` with its label over its
center, so that the label keeps the size of the text: the symbol of an
equation was scaled down with its node ([next#575](https://github.com/rel-int/discopy-next/pull/575)).
