`feedback.Diagram.feedback` and `trace` raise `AxiomError` on a wrong `dom`
or `cod` in the keyword form when the memory is not delayed, instead of
silently using the diagram's own boundary ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
