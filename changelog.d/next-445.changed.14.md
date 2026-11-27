Delaying a `Sum` or a `Bubble` works wherever delaying a box does: delay
distributes over a sum, so `(f + g).delay() == f.delay() + g.delay()`,
while a delayed bubble, circuit bubble or `egraph.Join` keeps its args
and carries a `time_step` of its own on a delayed boundary, so
`f.bubble().delay() != f.delay().bubble()` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
