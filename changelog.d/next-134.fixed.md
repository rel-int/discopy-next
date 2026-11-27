`utils.get_origin` of a union is the union of the origins, so that a
`python.Function` type-checks a value of `frozenset[int] | None`, e.g. a
channel of `Maybe` on a generic type ([next#134](https://github.com/rel-int/discopy-next/issues/134)).
