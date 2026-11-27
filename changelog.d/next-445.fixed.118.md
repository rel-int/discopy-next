`Circuit.measure` and `get_counts` without a backend accept a delayed circuit
and agree with the undelayed one, instead of raising `AxiomError`; `to_tk`
raises `NotImplementedError` on one, as on anything else it cannot translate
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
