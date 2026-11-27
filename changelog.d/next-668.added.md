`discopy.lean` compiles the categories of `discopy.abc` and their axioms to
Lean structures, read off the declarations by introspection, into
`discolean/DisCoLean/ABC.lean`; `discolean/` proves a compiled category is a
category of Mathlib and that strict premonoidal categories, Diagram(S)
included, satisfy the laws of `monoidal.Diagram` ([next#668](https://github.com/rel-int/discopy-next/pull/668)).
