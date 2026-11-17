> cubic P3: `rectangles` re-derives the box geometry that `Drawing.add_box_corners` computes; reuse it, e.g. `add_box_corners` on a `relabel_nodes()` copy, then `Drawing.rectangles()`.

- [ ] rectangles: Drawing.rectangles of a copy with its corners added, widened to the label
