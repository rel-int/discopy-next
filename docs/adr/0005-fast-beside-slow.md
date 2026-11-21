# 5. A fast specialised implementation lands beside the slow general one that checks it

## Context

A general construction, such as double-pushout rewriting on `Hypergraph` or the `Int`
construction, is slow, but it follows the definition. A specialised data structure, such as
Lafont's interaction nets or a token machine, is fast but is code of its own that can go wrong.
toumix, 2026-11-14, on [#542](https://github.com/rel-int/discopy-next/issues/542): "the fast
interaction net implementation also has its place in the library next to the slower but more
general DPO which can be used to validate it, same for message passing".

## Decision

A specialised implementation does not replace the general one. Both stay in the library, and a
law checks that they agree: the fast side is tested against the slow side on generated terms. The
specialised implementation is a representation beside the others, not a level of the hierarchy.

## Consequences

- Each fast implementation comes with a property test that relates it to the general one. For
  #542 the proposed law is that a net reduced on its own and the same net reduced by DPO as a
  hypergraph give the same hypergraph.
- The e-graph already follows it: `EGraph.stepping` states that a round on `egraph.Builder` and a
  pure `EGraph.step` reduce to the same hypergraph
  ([#586](https://github.com/rel-int/discopy-next/pull/586)).
- The general construction must stay fast enough to run inside the property tests on small
  terms.

## Status

Accepted, 2026-11-14. Interaction nets are not built yet: #542 is open.
