A hypergraph or a map of a level declares the categories of that level:
`Hypergraph[C]` and `CMap[C]` subclass the abstract base classes of `C` up to
`abc.HypergraphCategory`, its linear part left out, rather than
`abc.DaggerMonoidalCategory` for every `C` and `abc.CompactCategory` for
every map, so that `issubclass(Hypergraph[braided.Diagram],
abc.BraidedCategory)` and not `abc.SymmetricCategory`, and a functor's
`requires` is checked on the subscript. `abc.HypergraphCategory` is an
`abc.ClosedMarkovCategory`. A hypergraph refuses when it is built a wiring
that is no morphism of its level, e.g. `Hypergraph[symmetric.Diagram].copy`
or a crossing in `Hypergraph[monoidal.Diagram]`, which failed only in
`to_diagram` before. A Markov category that is not a hypergraph category,
e.g. `neural.Network`, keeps its merges as boxes, since a spider for both
copy and merge proved the Frobenius law: its hypergraphs are neither
comarkov nor dagger, and `Network` declares its merge laws failing. Below
symmetric categories, `Hypergraph.to_diagram` places a state where its
wiring puts it, so that a boundary-connected hypergraph needs no offset; an
offset only places a scalar ([next#783](https://github.com/rel-int/discopy-next/pull/783)).
