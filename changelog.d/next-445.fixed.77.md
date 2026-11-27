`para.Closed.curry` raises `ValueError` ("Cannot split n atoms off dom") for
`n < 0` or `n > len(dom)`, as `para.Compact.curry` and `para.Traced.trace`
do, instead of currying into the parameter wires and failing with an
`AxiomError` about an internal composite ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
