`optyx.channel` types keep their delay through `single()` and `double()`:
`qubit.delay().double()` is `(bit @ bit).delay()`, no longer `bit @ bit`
at time step zero ([next#667](https://github.com/rel-int/discopy-next/pull/667)).
