// The layout of the paper. `transpile.py` reads the calls defined here and
// writes the LaTeX of each one, with `preamble.tex` as its layout.

#let paper(title: none, authors: (), abstract: none, body) = {
  set document(title: title, author: authors)
  set page(paper: "a4", margin: 2.5cm, numbering: "1")
  set text(size: 10.5pt)
  set par(justify: true)
  set heading(numbering: "1.1")
  show link: set text(fill: rgb("#1f4e79"))
  show raw.where(block: true): block.with(
    width: 100%, fill: luma(245), inset: 8pt, radius: 2pt)
  show raw.where(block: true): set text(size: 8.5pt)
  align(center, text(size: 17pt, weight: "bold", title))
  align(center, authors.join(", "))
  v(1em)
  block(inset: (x: 2em), [*Abstract.* #abstract])
  v(1em)
  body
}

// The drawing of the Python value `name` of the snippets, which `build.py`
// writes to `figures/` once for Typst and once for TikZ.
#let drawing(name, caption: none) = figure(
  include "figures/" + name + ".typ", caption: caption)

// What the draft does not say yet.
#let gap(body) = block(
  width: 100%, inset: 8pt, radius: 2pt,
  stroke: (left: 2pt + rgb("#b85c00")), fill: rgb("#fff4e5"),
  [*Not written yet.* #body])
