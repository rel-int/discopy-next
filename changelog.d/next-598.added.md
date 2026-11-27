`discopy.drawing.nicer.nicer_layout` searches the geometry of a drawing:
`stretch` scales its column and row gaps and its box sizes through
`Drawing.relayout`, and a step is kept when it scores better and keeps the
`clearance` between boxes and from the inputs' and outputs' rows
([next#598](https://github.com/rel-int/discopy-next/pull/598)).
