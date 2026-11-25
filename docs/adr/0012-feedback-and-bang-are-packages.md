# 12. Feedback and bang are packages

## Context

Every level of the hierarchy from `balanced` and `pivotal` down was a feedback category: its
wires carried a time step and `trace`, `feedback` and `delay` were methods of every diagram.
`discopy.bang`, the free linear category, sat beside the hierarchy right above `monoidal`.
Joseph and Alexis ruled on 2026-11-25: "it's too messy to have every module be feedback/traced
by default, we should have a feedback package with submodules for balanced/symmetric/markov/
closed", and "same for bang, [...] it should be its own package with modules for everything
that makes sense to bang. feedback and bang shouldn't have to interact until we've figured out
a meaningful way to do so".

## Decision

- `discopy.feedback` is a package: the monoidal level in its `__init__`, then `balanced`,
  `symmetric`, `markov`, `comarkov` and `closed`, each subclassing the ordinary module of the
  same name and the feedback level above it. They are the only modules whose wires carry a time
  step and whose diagrams have `delay`, `feedback`, `trace`, `head` and `tail`.
- `discopy.balanced`, `symmetric`, `markov`, `comarkov` and `closed` are not feedback or traced.
  `abc.BalancedCategory` is no longer traced; `abc.BalancedTracedCategory` states the yanking that
  gives the twist, and `abc.RibbonCategory` extends it.
- `pivotal` and the levels below it, `ribbon`, `compact` and `frobenius`, stay traced and keep
  their time step: their trace is made of cups and caps, not assumed.
- `discopy.bang` is a package: the planar level in its `__init__` and `symmetric`, where the
  exchange of a banged wire is its swap.
- Feedback and bang share no class: no banged wire has a time step and no feedback diagram has
  a bang.

## Consequences

- The names that moved have no alias (ADR 1): `symmetric.Trace` is `feedback.symmetric.Trace`,
  `markov.FollowedBy` is `feedback.markov.FollowedBy`, `symmetric.Diagram.wait` is
  `feedback.symmetric.Diagram.wait`, and so on; `closed.Timed` is `feedback.closed.Timed`.
- A plain `markov` or `closed` diagram reads `str` and `repr` with `monoidal.Ty`, not
  `feedback.Ty`; closed terms, i.e. lambda terms, live in plain `closed` and do not delay.
  `closed.Diagram.to_compact` is `feedback.closed.Diagram.to_compact`, since it needs a trace.
- `egraph.Join` is a Markov bubble and no longer delays; `neural.Network` is traced through
  `feedback.markov` and `feedback.comarkov`.
- `bang` stops at `symmetric`: a Markov category copies and discards every wire already, so the
  contraction and weakening of a bang add nothing there, and a bang over a closed or compact
  category has to say how it meets the exponentials or the duals, which is a design of its own.
  A bang over those levels, and feedback of a banged diagram, wait on a later ruling.
- `Hypergraph.plug`, which rewriting calls, glues the hole into its context with
  `Hypergraph.glue`, not `trace`: the pushout of a convex match needs no trace in the category,
  so plain symmetric hypergraphs still rewrite.

## Status

Accepted 2026-11-25, carried out by the pull request that adds this record.
