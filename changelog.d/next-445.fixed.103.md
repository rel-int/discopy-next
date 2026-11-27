Mixed evaluation of a daggered or transposed pure box, e.g. `S.dagger()` or
`U.r`, is the doubled pure evaluation, by `channel.Functor` and by
`Circuit.eval(mixed=True, contractor=...)`, so circuits with a measurement
give the right probabilities ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
