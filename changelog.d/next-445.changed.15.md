A tensor or circuit box, and so a tensor bubble, counts its `time_step` in
`==` and `hash`, as a feedback box does: for circuit types,
`Box('f', x, y).delay() != Box('f', x.d, y.d)`, where they used to compare
equal though their resets differ ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
