`Circuit.eval(backend=..., measure_all=True)` and `Circuit.get_counts`
measure each open qubit into a fresh bit, so `eval` no longer raises
`IndexError` and a circuit with a `Bra` no longer post-selects on the wrong
qubit ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
