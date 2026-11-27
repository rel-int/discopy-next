Rotating or conjugating a `Ket`, `Bra`, `Bits`, `Digits`, `Discard` or
`MixedState` reverses its wires: `Ket(1, 0).r == Bra(0, 1)`,
`Ket(1, 0).conjugate() == Ket(0, 1)` and
`Discard(bit @ qubit).r == MixedState(qubit @ bit)` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
