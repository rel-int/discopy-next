`Hypergraph.make_bijective`, and so `to_diagram` over `compact`, raise
`AxiomError` (`NOT_FROBENIUS`) on a spider of three or more legs over a
category with no spiders, where they raised `AttributeError`; the unused
messages `MATRIX_TWO_DTYPES`, `HAS_NO_ATTRIBUTE` and
`COMPLEX_TYPE_HAS_NO_ATTR` are gone ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
