The TikZ backend sets a box label at 8 points and a wire label at 7, below
and to the right of its port as in Matplotlib, in the colour the backend
picks for it, and a unit of its picture is 8/12 inch, which fits the width
the layout measures for a label: every box fits its label, and pictures are
1.7 times larger. A wire of a drawing is named by its object's
`drawing_name` when it has one, and a `monoidal.Dim` is drawn as its
numbers, `4` for `Dim(4)` ([next#605](https://github.com/rel-int/discopy-next/pull/605)).
