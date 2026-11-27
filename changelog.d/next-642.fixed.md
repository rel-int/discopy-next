A label between dollars is LaTeX in the Typst backend too, as in
matplotlib and TikZ: `drawing.typst_ast.latex_math` writes it in Typst
mathematics from a table of the commands it knows and raises on any other,
so `$\mapsto$` draws in Typst, where it stopped on `unknown variable`, and
`$<=$` is two symbols rather than `≤` ([next#642](https://github.com/rel-int/discopy-next/pull/642)).
