`optyx.channel.Diagram.from_tket` and `optyx.qubits.Circuit` convert a
`pytket.Circuit` to an open diagram on `qubit ** n @ bit ** m`, keeping its
unitary (global phase included) and routing non-adjacent gates and
measurements; `QubitChannel.from_discopy` gains `U1`, `Ry`, `CRy`, `CY`,
daggered gates, `Discard` and `Measure` ([next#296](https://github.com/rel-int/discopy-next/issues/296)).
