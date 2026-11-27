`optyx.compiler`: a `Fusion` compared to anything but a fusion is unequal
rather than an `AttributeError`; `path_cover.find_min_path_cover` covers a
graph with one node, an edgeless graph and the empty graph, searching up to
as many paths as the graph has nodes rather than one fewer; and the SEMM
decompiler refuses three invalid instruction streams it used to accept — an
instruction before the first `next_node`, a node measured twice when a delay
records one of the two in the future, and a delayed fusion no photon comes
back for ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
