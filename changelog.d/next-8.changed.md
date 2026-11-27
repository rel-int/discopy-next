A `Matrix` with no dtype reads it off its array without copying it, so it
copies once, as a `Matrix[dtype]` does ([next#8](https://github.com/rel-int/discopy-next/issues/8)).
