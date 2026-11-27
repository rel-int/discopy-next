What the property matrix of [next#1](https://github.com/rel-int/discopy-next/pull/1) found, each with its test:
`Feedback.dagger` and `rotate` raise `AxiomError`, as either would feed
the memory back in time; `para.Compact.curry` keeps the coparameters and
`para.Closed.curry` raises `ValueError` on them; a layer drops a spider
with one leg on each side, e.g. `Spider(1, 1, x)`, for its type; abstract
diagrams copy and discard with a `Function` of their own; the hypergraph of
a traced copy of a wire whose dagger is not itself, e.g. a
`Dims(Dim(2, 3))`, is built; and the strings of delayed boxes, structural
boxes and adjoint wires read back, their `atoms` being the generators they
name, as do those of a hypergraph; a Markov hypergraph whose spider has no
input, i.e. a copy traced back into itself, decodes to that trace.
