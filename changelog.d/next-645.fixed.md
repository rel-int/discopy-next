The Typst backend fills the coloured regions of a drawing and the shapes of
a ribbon, as TikZ does: `Typst.draw_filled_shape` is a CeTZ `merge-path`
and `Typst.draw_region_cell` fills a cell through it, so the frame of an
e-graph's `Join` and the band of a ribbon show ([next#645](https://github.com/rel-int/discopy-next/pull/645)).
