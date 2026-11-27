`tensor.Functor` contracts a `frobenius.Hypergraph` as it stands,
`Functor.spider_operands` writing its Einstein notation: one index per
spider however many ports it has, one array per box, the boundary as free
indices and a spider on no port summed alone, so that it counts its type.
With NumPy, the greedy path bounds intermediates by the largest operand,
which forbids every pairwise step when an index is on many arrays. A
hypergraph is contracted without that bound: the 4×4 sudoku, as 16 spiders
and 12 boxes, counts its 288 grids in milliseconds, where its diagram raises
`iterator is too large` ([next#554](https://github.com/rel-int/discopy-next/pull/554)).
