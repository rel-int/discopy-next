Optyx reads a foliated diagram one box at a time: `Diagram.sample`, `dilate`,
`get_kraus`, `explode_channel` and `BitControlledGate` no longer raise
`ValueError` on a layer that holds several boxes ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
