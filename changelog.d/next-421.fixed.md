`optyx.core.zx.Z(0, 0, phase)` is the scalar `1 + exp(2i pi phase)`, the sum
of its amplitudes, which is what a spider with no index left to carry them
is and what pyzx gives: it used to be a legless `zw.ZBox` carrying both
amplitudes, so `qubits.Z(0, 0)`, `qubits.X(0, 0)` and `Z(0, 0, .25)` raised
`cannot reshape array of size 2 into shape ()` rather than evaluating
([next#421](https://github.com/rel-int/discopy-next/issues/421)).
