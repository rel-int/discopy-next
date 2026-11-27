# Dataflow is a package

## Context

The head and tail of a stream and its followed-by, Lucid's `fby`, were part of
`discopy.feedback`: every wire had `head`, `tail` and `is_constant`, every diagram `head`, `tail`
and `followed_by`, and `feedback.structural_boxes` and `structural_wires` generated a `Head`,
`Tail`, `FollowedBy`, `HeadOb` and `TailOb` at every level down to `frobenius`, `quantum.circuit`
and `optyx`, which [Traced levels with duals](2026-11-26-traced-levels-with-duals.md) carried down. Their only
interpreted uses are in streams, the Fibonacci sequence and the random walk among them, and two
of their cases were silent: `Nat`, `Dim` and `Dims` were their own head and tail, and a functor
into a category without streams sent `f.head` to a bubble. Alexis ruled on 2026-11-27 that they
leave `discopy.feedback` for their own module, `discopy.dataflow`, after "Monoidal Streams for
Dataflow Programming" (Di Lavore et al. 2022), with `discopy.stream` as their semantics, then that
`dataflow` is a package parallel to `feedback` with two levels, symmetric and Markov.

## Decision

- `dataflow.symmetric` is the base level, over `feedback.symmetric`: monoidal streams are defined
  over symmetric categories, and none of the primitives copies. Its classes hold the structure,
  `Wire` (`head`, `tail`, `is_constant`), `Ty`, `Diagram` (`head`, `tail`, `followed_by` and
  their factories), `HeadOb`, `TailOb`, `Head`, `Tail`, `FollowedBy` and `Functor`, and the
  package's `__init__` only documents the package and imports its two levels. A level above it,
  e.g. `dataflow.markov` over `feedback.markov`, is one class per class of the base level mixed
  into its feedback level, written out, nothing generated.
- A dataflow functor sends the primitives to those of its codomain, e.g. `stream.Stream`, and
  raises on a codomain without them, whether its map is a mapping or a callable: they are the
  structure of the category, not generators. The hypergraph of a dataflow diagram has them, so
  `to_hypergraph` and `Equation` go through: `dataflow.symmetric.Hypergraph`, the
  `hypergraph_factory` of both levels, has the head and the tail of the diagram it decodes to as
  boxes, the followed-by as a box and a delay of each box and spider.
- Their laws are `abc.DataflowCategory` and `abc.DaggerDataflowCategory`, stated once as on
  every other structure: head and tail are monoidal endofunctors, `H∘H = H`, `T∘H = 1`,
  `H∘D = 1` and `T∘D = Id` are equations of functors checked on an object and a morphism at once,
  and the followed-by is natural, unitary and the identity on a delay. The free dataflow diagrams
  satisfy them up to their streams, on the diagrams of the symmetric rules: in a symmetric
  category the head of a feedback has no stream, as the memory its first step writes cannot be
  discarded.
- No other level has them, and `feedback` and `dataflow` keep the same direction of import as
  `feedback` and the plain levels: `dataflow` imports `feedback`, never the converse.

## Consequences

- The names that moved have no alias ([Renames take no alias](2026-11-20-renames-take-no-alias.md)): `feedback.FollowedBy` is
  `dataflow.symmetric.FollowedBy` or `dataflow.markov.FollowedBy`, and the `Head`, `Tail`,
  `FollowedBy`, `HeadOb` and `TailOb` of the other levels are gone, as are
  `feedback.structural_wires` and the heads of `quantum.circuit` and `optyx` wires. A tree or
  pickle saved with them does not load.
- `Nat`, `Dim` and `Dims` have no head or tail.
- Optyx as a dataflow level, with its initial states written with `fby`, waits on a pull request
  of its own.

## Status

Accepted, 2026-11-27. [#768](https://github.com/rel-int/discopy-next/pull/768), stacked on #744,
carries it out.
