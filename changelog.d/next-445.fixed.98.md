`optyx.qubits.X(1, 1, .25).dual_rail()` is `BS` times `exp(-i pi / 4)`, and
`X(1, 1, .75)`, also written `-.25`, is `BS.dagger()` times `exp(i pi / 4)`,
where the old code returned the wrong beam splitter or raised
`NotImplementedError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
