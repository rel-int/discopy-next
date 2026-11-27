`optyx` channels and gates load their own tree: `channel.Measure`,
`Encode` and `Discard` and their `photonic` and `qubits` subclasses from
their domain, and `photonic.Phase`, `BBS`, `TBS` and `MZI` from their data
and flags, so `loads(dumps(...))` no longer raises on them ([next#739](https://github.com/rel-int/discopy-next/pull/739)).
