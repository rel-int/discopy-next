The `inside` of a composition of `python.Function` is a
`python.function.Composition` of its `factors`, extended when composed
again, so that a long composition built one function at a time is called
by a loop rather than by as many nested calls ([next#518](https://github.com/rel-int/discopy-next/pull/518)).
