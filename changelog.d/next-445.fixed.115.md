A pure circuit `Bubble` evaluates under `mixed=True` as the doubling of its
pure evaluation (single for a classical one), instead of raising
`AttributeError` on `is_classical` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
