Equal hypergraphs hash equal when their boxes are equal but print
differently (`data=1` and `data=1.0`): the fast path of
`Hypergraph.__hash__` hashes the boxes, not their `repr`, so a box with
unhashable data makes its hypergraph unhashable, as it does its diagram
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
