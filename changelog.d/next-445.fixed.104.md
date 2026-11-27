`to_tk` emits a transposed `Y` with its sign, since `Y.r == -Y`: `CY.l` and
`CY.r` go to tket as `CY` followed by `Z` on the control, so tket samples the
same outcomes as `eval` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
