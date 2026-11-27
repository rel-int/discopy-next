`matrix.Contractor`, a backend whose `einsum` contracts the arrays of
another backend, with `OptEinsum` and `Quimb`, which takes `cotengra`
path optimizers as `optimize` and contracts approximately past a
`max_bond`, chosen by the `contractor` parameter of `tensor.Functor`,
`tensor.Diagram.eval` and `quantum.Circuit.eval`; the Einstein notation is
`tensor.Functor.operands`, and the contraction dependencies are the new
`tensor` extra ([#533](https://github.com/discopy/discopy/pull/533), [next#495](https://github.com/rel-int/discopy-next/pull/495)).
