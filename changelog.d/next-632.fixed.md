An array holding `nan` or `inf` reads back from its `repr`: `Matrix`,
`Tensor`, `tensor.Box`, `optyx.core.diagram.Box` and `photonic.Gate`
print them as `float('nan')`, `float('-inf')` or `complex('(1+nanj)')`,
spelled by `matrix.repr_nonfinite`, rather than the bare tokens `eval`
cannot name, and a `tensor.Box` prints its data as a list, not an
`array(...)` ([next#632](https://github.com/rel-int/discopy-next/issues/632)).
