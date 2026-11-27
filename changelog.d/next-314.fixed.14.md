Every code block of `discopy/optyx/README.md` runs, and `test/optyx/readme.py`
checks that it does, the way `test/readme.py` checks the root README: the
snippets import `Channel`, `qubits.H` and `photonic.FusionTypeII`, call
`classical.Add` where `AddN` does not exist, call `Ket` as they import it,
define the `state_array` they send through the perceval processor, read
graphix's statevector off the backend as graphix 0.4 leaves it, and give
`np.isclose` and `np.allclose` no third positional argument, so that the
teleportation and distinguishability assertions check what the prose around
them claims: `np.isclose` was reading a `3` as a 300% relative tolerance, and
`np.allclose` a doubled identity channel as one ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
