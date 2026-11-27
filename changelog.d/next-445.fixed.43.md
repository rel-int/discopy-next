Products and exponentials print their name only when it is first read, so
nested `Tuple`s and exponentials build in linear time, and a thousand nested
`Tuple`s no longer hit the recursion limit ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
