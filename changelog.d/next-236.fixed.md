`cat.Functor` reads the image of a daggered box off `ar_map` when the
codomain has no dagger to take, instead of daggering the image: a functor
into `stream.Stream` or `python.Function` used to raise an `AttributeError`,
so `double` and `unroll` stopped commuting on a loop whose doubling makes a
daggered box ([next#236](https://github.com/rel-int/discopy-next/issues/236)).
