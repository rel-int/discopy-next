`optyx.core.path.Matrix` survives `dumps` and `loads`: its tree names
`creations`, `selections`, `normalisation` and `scalar` beside the array,
and `from_tree` reads `dom` and `cod` as the mode counts they are. It
inherited `matrix.Matrix.to_tree`, which writes the array, `dom` and `cod`
alone, so a matrix with a creation or a selection raised on the array's
shape, the array being the underlying one, and a matrix without came back
as a different map with its normalisation and its scalar lost. A
`path.Sum` of such terms is fixed with them ([next#698](https://github.com/rel-int/discopy-next/pull/698)).
