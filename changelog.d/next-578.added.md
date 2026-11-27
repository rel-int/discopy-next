`Drawing.layout`, the geometry of a drawing as a vector: the coordinates
of its nodes, box corners included. `Drawing.relayout` takes a vector back
to a drawing and refuses one that `Drawing.is_feasible` says draws another
diagram (an edge turned, other wires crossing, boxes overlapping, a port
outside its box) ([next#578](https://github.com/rel-int/discopy-next/pull/578)).
