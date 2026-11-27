`neural.propagation`, belief propagation: `sum_product` is the
sum-product algorithm on a factor table as a network with a trace, from
the tensor of each factor to the belief of each node, its traced legs the
messages and a round its `exclusive`, `reply` and `belief` boxes;
`marginals` contracts the same tensors, and on a factor table whose
incidence graph is a tree the two agree after `radius` rounds
([next#520](https://github.com/rel-int/discopy-next/pull/520)).
