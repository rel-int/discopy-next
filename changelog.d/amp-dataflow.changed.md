The head, the tail and the followed-by of streams leave `discopy.feedback`
for a package of their own, `discopy.dataflow`, after "Monoidal Streams
for Dataflow Programming": `dataflow.symmetric`, the level over
`feedback.symmetric`, holds them, the wire with its `head`, `tail` and
`is_constant`, the type, the diagram with its `head`, `tail` and
`followed_by`, the `Head`, `Tail` and `FollowedBy` boxes and the functor
that maps them, and `dataflow.markov` mixes them into `feedback.markov`.
Their laws are `abc.DataflowCategory` and `abc.DaggerDataflowCategory`:
head and tail are monoidal endofunctors with `H∘H = H`, `T∘H = 1`,
`H∘D = 1` and `T∘D = Id`, and the followed-by is natural, unitary and the
identity on a delay, which the property matrix checks of dataflow diagrams
up to their `to_stream`. `HeadOb`, `TailOb`, `Head`, `Tail`, `FollowedBy` and
`Wire.is_constant` go from `feedback` and from every level that generated
them, with no alias, and so do `feedback.structural_wires` and the heads of
`quantum.circuit` and `optyx` wires: a tree or pickle saved with them does
not load. A dataflow functor raises on a codomain without a head, a tail
and a followed-by of its own, whatever its map, where a feedback functor
made up a bubble: the hypergraph of a dataflow diagram has them, so
`to_hypergraph` and `Equation` go through. `Nat`, `Dim` and `Dims` lose
the `head` and `tail` that were themselves, and `stream.Stream.copy`
discards when copying zero times ([next#768](https://github.com/rel-int/discopy-next/pull/768)).
