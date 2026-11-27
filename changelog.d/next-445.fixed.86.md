`optyx.core.diagram.Scalar(s).inflate(d)` is `Scalar(s)` rather than
`s ** d`, and a `Sum` inflates term by term, so a core diagram holding a
scalar or a sum keeps its amplitude when inflated ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
