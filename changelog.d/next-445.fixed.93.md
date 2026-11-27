`subs` and `lambdify` substitute the parameter of optyx's `qubits.Scalar`,
`photonic.Gate`, `PhotonLoss`, `PhaseShiftDR` and `classical.Z` and `X`,
which report it in `free_symbols`: they used to leave the symbol in place,
and their gradient was silently zero. A `Gate` with a symbolic matrix and no
`data` is substituted too ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
