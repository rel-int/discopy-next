A selection that no photon number can satisfy is an amplitude of zero, where
`PercevalBackend` used to raise `TypeError` from the SLOS backend for an
effect outside the Fock space it enumerates: `Create(1) >> Select(2)` and
`Create(1, 0) >> BS >> Select(2, 0)` are zero, as every other backend says,
rather than a crash on `amps` and `1.0` and `0.5` on `probs` ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
