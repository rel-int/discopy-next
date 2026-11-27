`optyx.core.zw.ZBox(0, 0, amplitudes)` reads the first of its amplitudes,
as its truncation already did for an infinite amplitude list and as `zw`'s
own axioms write a scalar: the list branch was missing that index, and a
legless box carrying more than one amplitude raised on the reshape
([next#421](https://github.com/rel-int/discopy-next/issues/421)).
