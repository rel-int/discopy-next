Composing a zero (an empty `Sum`) with an arrow it does not compose with now
raises `AxiomError`, on either side, rather than returning an ill-typed zero
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
