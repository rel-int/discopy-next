`Sum` keeps its terms as a tuple, so `Sum([f, g]) == Sum((f, g)) == f + g`
and is hashable, and a zero gradient `Sum([], dom, cod)` equals
`Sum((), dom, cod)` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
