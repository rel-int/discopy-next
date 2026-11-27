`CMap.match` sends a pattern's straight wires only to the wires
`CMap.beside` its image, so matching a box with a few wires beside it no
longer takes time exponential in their number (about 35 s down to 0.2 s at
five wires), with the same matches ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
