`Matrix.resolvent`, the inverse of `id - D` over a field, which a
`Tensor` inherits, and with it
`Matrix.trace` over `float` and `complex`: the trace of Haghverdi and
Scott, `A + B (id - D)^-1 C`, a partial trace raising `ValueError` when
`id - D` is singular (Malherbe, Scott and Selinger, arXiv:1107.3608,
Definition 3.11). `bool` keeps the Kleene star, any other dtype raises
`TypeError`; `matrix.is_field` says which dtypes are a field, in any
backend ([next#550](https://github.com/rel-int/discopy-next/pull/550)).
