Calling a `lambek.cfg.Tree` on arguments whose types do not tensor to its
`dom` (all identities, an `Id` root, too many or too few arguments) raises
`AxiomError`, where it used to return an ill-typed or truncated tree
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
