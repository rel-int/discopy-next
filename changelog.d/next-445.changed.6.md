Composing a `DualRail` with any functor but the identity, on either side and
at every level (balanced, symmetric, ribbon, compact), raises `AxiomError`
saying to apply them in turn, `E(D(f))`, set by a new `Functor.composes`
that is false on `DualRail`: `D >> F` used to raise an unrelated `TypeError`,
and `F >> D` returned a functor that disagreed with `D(F(-))` on cups and
raised on twists, braids and swaps ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
