A hypergraph or a map of a level declares the categories of that level:
`Hypergraph[C]` and `CMap[C]` subclass the abstract base classes of `C` up to
`abc.HypergraphCategory`, its linear part left out, rather than
`abc.DaggerMonoidalCategory` for every `C` and `abc.CompactCategory` for
every map, so that `issubclass(Hypergraph[braided.Diagram],
abc.BraidedCategory)` and not `abc.SymmetricCategory`, and a functor's
`requires` is checked on the subscript. `abc.HypergraphCategory` is an
`abc.ClosedMarkovCategory`. Below symmetric categories,
`Hypergraph.to_diagram` places a state where its wiring puts it, so that a
boundary-connected hypergraph needs no offset; an offset only places a
scalar ([next#783](https://github.com/rel-int/discopy-next/pull/783)).
