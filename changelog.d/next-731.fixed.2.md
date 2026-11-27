`optyx.core.diagram`, `optyx.core.zx` and `optyx.classical` types and boxes
load their own tree (`Mode`, `Bit`, `Box` with `array` or `is_dagger=None`,
`Scalar`, `DualRail`, `PhotonThresholdDetector`, `EmbeddingTensor`, `zx.H`,
`Spider`, `And`, `Or` and the value-initialised `classical` boxes), while
`ClassicalFunction` and `BitControlledPhaseShift` raise `TypeError` on
`to_tree` since JSON cannot hold a Python callable ([next#731](https://github.com/rel-int/discopy-next/issues/731)).
