The kappa calculus, the first-order half of the lambda calculus: each
module below the closed categories has terms of its own, `Variable`,
`Constant`, `Application` and `Kappa`, with `let` for the redex. In `cat`
a term has one variable and is a sequence of boxes: `Arrow.to_term` reads
an arrow, `eval` gives it back and terms are equal up to alpha, beta and
eta exactly when their arrows are. In `monoidal` a term is planar and
linear, with a context of variables, a stack of inputs and one output:
`Diagram.to_term` reads a diagram with exactly one output as a term with a
`let` for each box but the last, which gives the output, and refuses any
other, and `normal_form` inlines the boxes with one output, up to
interchange, which `monoidal.InterchangeEquation` decides for small
diagrams by a search of the orbit under interchangers. In `symmetric` a
`Kappa` binds its variables in any order: `Diagram.to_term` reads a swap
as the order of the variables and `eval` gives the diagram back up to the
axioms of symmetric categories. In `markov` a variable occurs any number
of times: an `Application` copies the variables its function and its
argument share and a `Kappa` discards the ones its body does not use. No
discard is assumed natural, so `normal_form` only inlines a variable used
exactly once and keeps the statement of one copied or discarded. The
modules that add no way to bind a variable, `spacial`, `braided`,
`feedback`, `balanced` and `comarkov`, get terms of their own from
`cat.kappa_calculus`, with their structural boxes as constants. A type
applied to a Python function is its kappa abstraction, and the laws are
cells of the property matrix ([next#658](https://github.com/rel-int/discopy-next/pull/658)).
