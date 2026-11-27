`Circuit.measure` measures every open qubit when it evaluates as a channel,
instead of discarding it, so `(Ket(0) >> H).measure(mixed=True)` is
`[.5, .5]` and `((Ket(0) >> X) @ Bits(1)).measure()` is the joint
distribution `[[0, 0], [0, 1]]` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
