`tensor.Diagram.to_quimb` is built on the combinatorial map, through
`tensor.Functor.to_quimb`: swaps, cups and caps are wiring instead of
dense tensors, the network is deterministic and `dtype` takes effect
([#533](https://github.com/discopy/discopy/pull/533), [next#495](https://github.com/rel-int/discopy-next/pull/495)).
