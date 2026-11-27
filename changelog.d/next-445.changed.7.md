`monoidal.Wire` equality and hash count `is_dagger`, and a daggered wire
prints with `[::-1]` as a daggered box does: `Ty('x')[::-1] != Ty('x')`,
a reversed coloured wire no longer equals a fresh one with its colours
swapped, and a wire and its dagger, which a functor may send to different
images, have different heads and tails. Wires and types break ties in
their order by `repr`, so of two unequal types exactly one is less than
the other ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
