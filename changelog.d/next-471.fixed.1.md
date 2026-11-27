`Hypergraph.trace_backward_wires`, and so `make_causal` and `make_cocausal`,
cuts its wires in a loop rather than one frame of recursion per cut, so a
graph with a thousand of them no longer raises `RecursionError` and no
longer holds every intermediate hypergraph alive at once. It is as fast as
it was: four fifths of the time is `explicit_trace`, which this leaves
alone ([next#471](https://github.com/rel-int/discopy-next/issues/471)).
