`Matrix.__le__`, the pointwise order on parallel matrices, which a
`Tensor` inherits: over `bool` it is the inclusion of relations, over the
complex numbers it raises `TypeError`. `testable.Inequation`, an `Equation`
whose terms are compared with `<=`, so that an axiom states an inclusion
and the property matrix checks it as it does an equation ([next#566](https://github.com/rel-int/discopy-next/pull/566)).
