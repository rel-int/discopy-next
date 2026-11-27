`neural.network.tape` and `Box.tape`, the outputs of a layer and its
pullback, as `jax.vjp` returns them: the `vjp` of the box at its inputs
or the reverse mode of PyTorch or JAX. A `Lens` keeps the pullbacks as
its residual, so that its `put` runs each layer forward once rather than
once per layer after it, `n` forward calls for a chain of `n` layers
where they were `n(n-1)/2` ([next#519](https://github.com/rel-int/discopy-next/pull/519), fixes [next#516](https://github.com/rel-int/discopy-next/issues/516)).
