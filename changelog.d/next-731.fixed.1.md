Every remaining box of `optyx.photonic` writes and reads its own tree:
`Swap` from `left` and `right`, `Create` and `Select` from `photons`,
`Create` and `DualRail` with their `internal_states`, `Gate` from its
`array`, `Scalar`, `PhaseShiftDR` and `PhotonLoss` from their `data`,
`PhotonThresholdMeasurement` from its domain, and `NumOp` and `HadamardBS`
with no argument, so `loads(dumps(...))` round-trips across `optyx.photonic`
([next#731](https://github.com/rel-int/discopy-next/issues/731)).
