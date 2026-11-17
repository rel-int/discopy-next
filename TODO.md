> Codex P2: for a drawing returned by `Drawing.relayout`, the rendered box bounds are its corner positions, but `rectangles` rebuilds them from `box.height`, e.g. a layout scaled by 0.1 gets false label overlaps.

- [x] rectangles read a box's corners when the drawing has them, widened to its label; test on a relaid-out drawing
