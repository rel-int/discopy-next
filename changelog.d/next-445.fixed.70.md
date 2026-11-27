`Stream.delay` works on streams whose memory starts non-empty, e.g.
`Stream.sequence(name, x, y, m)` and the tail of a feedback, instead of
raising `AxiomError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
