`drawing.picture.decode` and `read`: a picture's pixel classes back to
its diagram. `regions` splits them into boxes, wires and labels, `attach`
hangs each label on its wire, `refine` writes the names and types read
back, and `scan` picks the smallest module whose primitives appear, at
least compact when a type is an adjoint ([next#601](https://github.com/rel-int/discopy-next/pull/601)).
