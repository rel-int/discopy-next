The dagger of a drawing with two or more boxes has a graph of its own, not a
frozen networkx view, so that `add_nodes`, `add_edges` and `add_box_corners`
work on it ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
