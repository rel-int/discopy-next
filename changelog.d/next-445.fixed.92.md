`grad` on an `optyx.channel` box that has free symbols and no gradient of its
own (`qubits.Z`, `X`, `Scalar`, `photonic.BBS`, `Gate`, a `CQMap` or a
feedback loop) raises `NotImplementedError` instead of recursing forever, and
still gives zero for a variable the box does not depend on; the
`free_symbols` of a `Channel` include those of its Kraus map and of a `CQMap`
those of its density matrix, so a box built from a symbolic diagram raises
too, where it gave a false zero ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
