`drawing.PlaneGraph` prints the adjacency of its graph rather than the
graph's address, so a `Drawing` reads back from its representation and
prints the same one for equal drawings: its nodes and their successors
are sorted by `Node.index`, and an adjacency, unlike a list of edges,
keeps a node that has none ([next#623](https://github.com/rel-int/discopy-next/pull/623)).
