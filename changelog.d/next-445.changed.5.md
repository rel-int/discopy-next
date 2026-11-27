`classical.BitControlledGate` prints the call to its own constructor,
`BitControlledGate(gate, default)` with `is_dagger` and `classical` when
set, so `eval(repr(x)) == x` holds for it and its dagger. It keeps its
gates as `data`, with the identity standing for a default not given, so
two boxes with different gates differ. Before, it printed the private
`_BitControlledSingleBox` with `Channel`'s arguments and compared equal for
any gates of the same type ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
