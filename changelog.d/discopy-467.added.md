A Typst drawing backend, `drawing.backend.Typst`: `Diagram.to_typst` and
`Drawing.to_typst` return a `drawing.typst_ast.Document`, whose `render`
is the Typst source of a CeTZ canvas, and `draw(format="typst")` compiles
it to SVG with the optional `typst` extra, or writes the source to a path
ending in `.typ` ([#467](https://github.com/discopy/discopy/pull/467), [next#565](https://github.com/rel-int/discopy-next/pull/565)).
