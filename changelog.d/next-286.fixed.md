`Hypergraph.is_convex` stops at the last box that writes into the image
instead of walking every box downstream of the match, wherever the boxes of
the host are topologically ordered, which `is_topologically_ordered` now
caches and a `Rewriter` declares false: matching a pattern over a host used
to cost the size of the host on every homomorphism ([next#286](https://github.com/rel-int/discopy-next/issues/286)).
