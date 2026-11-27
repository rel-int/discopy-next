`Drawing.bubble` takes a `dom` and a `cod` of plain types, e.g. symmetric,
rigid, compact or frobenius ones, casting them with `to_drawing`: it used to
leave nodes with no position when their length matched the inside's, so that
drawing raised `KeyError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
