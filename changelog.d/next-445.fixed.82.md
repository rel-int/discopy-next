`optyx.core.zx.ZBox` exports to PyZX as a `Z_BOX` vertex labelled with its
amplitude, and `from_pyzx` reads that vertex back as a `ZBox`, where
`to_pyzx` made it an X spider of phase twice the amplitude, so that
`ZBox(1, 1, .5)` became a NOT gate ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
