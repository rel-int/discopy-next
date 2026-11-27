`optyx.qubits.Circuit` converts discopy's `CRx`, `CU1` and controlled gates
at any distance to the right unitary, converts `SWAP` and `Controlled(U1)`
instead of raising `KeyError`, and raises `NotImplementedError` on a mixed
scalar instead of squaring it ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
