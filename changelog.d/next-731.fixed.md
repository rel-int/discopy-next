Every remaining box of `optyx.qubits` loads its own tree: `Z` and `X` from
their boundary and `data`, `H` with no argument, `Scalar` from `data`,
`BitFlipError` and `DephasingError` from `prob`, `Ket` and `Bra` from
`value` and their boundary, and `QubitChannel` from `kraus` and `env`
([next#731](https://github.com/rel-int/discopy-next/issues/731)).
