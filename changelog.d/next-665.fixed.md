`qubit.head`, `qmode.head` and `mode.head` raised, and so did `mode.delay()`:
the wires of `quantum.circuit`, `optyx.channel` and `optyx.core.diagram` now
have heads and tails of their own, and the head of a qubit is a quantum wire of its
dimension (`quantum.circuit.Wire.is_quantum`), so that the initial state of
a loop can be written `s.head @ m.d >> FollowedBy(m)` on circuits and
channels ([next#665](https://github.com/rel-int/discopy-next/pull/665)).
