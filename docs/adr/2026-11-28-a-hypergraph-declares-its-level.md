# A hypergraph declares its level

## Context

Every functor class declares in `requires` the abstract base class its codomain must subclass.
`Hypergraph[C]` declared `DaggerMonoidalCategory` whatever `C` was, and `CMap[C]`
`CompactCategory`, so the functor `from_diagram` builds into them was rejected at every level
that is not monoidal, and `symmetric.CMap` declared 45 compact laws inapplicable. Dropping to a
monoidal functor would make spiders, cups and copies boxes, which loses the faithfulness the
hypergraph is for. Alexis ruled on 2026-11-28: "yes they should declare their level but the main
issue is the contravariance: monogamous acyclic hypergraphs are a subclass of hypergraphs but the
symmetric monoidal categories they represent are a superclass of hypergraph categories", then, on
the offsets of a planar hypergraph, "for boundary connected the offset can be computed from the
structure alone", and, on a category with copies and merges but no Frobenius law, "merge is a box
copy is wiring".

## Decision

- Subclassing follows the structure, not the terms. `Hypergraph[C]` and `CMap[C]` subclass the
  abcs of `C` up to `HypergraphCategory`, set by `NamedGeneric` when the subscript is built
  (`level_bases`, `hypergraph.level`). No subscript subclasses another.
- The inclusion of terms is checked on the instance. `Hypergraph.scan_level`, called by
  `__init__` unless `_scan=False`, refuses a wiring that is no morphism of the level: spiders with
  two ports below rigid, one producer below Markov, one consumer below comarkov, monogamous
  otherwise; cycles only with a symmetric trace or along cups; planarity below symmetric.
- The linear part of a level is left out, since no wiring carries a bang: `from_diagram` walks to
  the functor whose `requires` the subscript satisfies.
- A Markov category that is not a hypergraph category copies on its spiders and keeps its merges
  as boxes: spiders for both would prove the Frobenius law. Its hypergraphs are neither comarkov
  nor dagger.
- A closed level that is not rigid keeps its curry as a box and declares the currying laws failing.
- Below symmetric, `to_diagram` places a state where the faces of its map put it; an offset only
  places a scalar.

## Consequences

- `abc.HypergraphCategory` is an `abc.ClosedMarkovCategory`, and `DisCoLean/ABC.lean` follows.
- A wiring off its level fails when it is built, not in `to_diagram`, e.g.
  `Hypergraph[symmetric.Diagram].copy(x)`.
- `Hypergraph.glue` and `EGraph.reduced` build with `_scan=False`: the first may close a cycle that
  a plug opens again, the second stores egg's tables, which are not left-monogamous.
- `neural.Network` loses its `spider_factory` and declares its merge laws failing, compared on the
  nose.
