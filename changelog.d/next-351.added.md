The constructor arguments the optyx boxes were not keeping, as public
attributes, so that they can print themselves: `qubits.Ket.value`,
`qubits.Bra.value`, `qubits.BitFlipError.prob`, `qubits.DephasingError.prob`,
`photonic.Create.internal_states`, `photonic.DualRail.internal_states`,
`photonic.PhaseShiftDR.phase`, `classical.Scalar.value`,
`classical.PostselectBit.bits`, `classical.PostselectDigit.digits`,
`classical.And.n`, `classical.Z.phase`, `classical.X.phase`,
`classical.ClassicalFunction.function` and
`classical.BitControlledPhaseShift.function` ([next#351](https://github.com/rel-int/discopy-next/issues/351)).
