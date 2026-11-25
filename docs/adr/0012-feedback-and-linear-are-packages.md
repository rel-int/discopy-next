# 12. Feedback and linear are packages

## Context

Every level of the hierarchy from `balanced` and `pivotal` down was a feedback category: its
wires carried a time step and `trace`, `feedback` and `delay` were methods of every diagram.
`discopy.bang`, the free linear category, sat beside the hierarchy right above `monoidal`.
Joseph and Alexis ruled on 2026-11-25: "it's too messy to have every module be feedback/traced
by default, we should have a feedback package with submodules for balanced/symmetric/markov/
closed", and "same for bang, [...] it should be its own package with modules for everything
that makes sense to bang. feedback and bang shouldn't have to interact until we've figured out
a meaningful way to do so". Alexis then: "let's call it linear instead of bang so we have
discopy.linear.cat with digging and dereliction, discopy.linear.monoidal with contraction and
weakening", "we should also add a GoISituation abstract class in abc", and on where the two
meet, "maybe that should go to interaction?" and "the triple should be a single abstract class
which subclasses linear and feedback with extra methods for the reflexive object".

## Decision

- `discopy.feedback` is a package: the monoidal level in its `__init__`, then `balanced`,
  `symmetric`, `markov`, `comarkov` and `closed`, each subclassing the ordinary module of the
  same name and the feedback level above it. Of the levels from `balanced` to `closed`, they are
  the only ones whose wires carry a time step and whose diagrams have `delay`, `feedback`,
  `trace`, `head` and `tail`; `pivotal` and the levels below it have them too, see below.
- `discopy.balanced`, `symmetric`, `markov`, `comarkov` and `closed` are not feedback or traced.
  `abc.BalancedCategory` is no longer traced; `abc.BalancedTracedCategory` states the yanking that
  gives the twist, and `abc.RibbonCategory` extends it.
- `pivotal` and the levels below it, `ribbon`, `compact` and `frobenius`, stay traced and keep
  their time step: their trace is made of cups and caps, not assumed.
- `discopy.linear` is a package: `linear.cat`, the comonad of `abc.ComonadCategory`, i.e. the
  bang, dereliction and digging, which needs no tensor; `linear.monoidal`, the free
  `abc.LinearCategory`, which adds contraction and weakening, and the laxator and exchange,
  which use the tensor. The exchange makes a banged object central and needs no symmetry, so
  there is no `linear.symmetric`. The method stays `bang()`.
- `abc.GoISituation` (Abramsky, Haghverdi and Scott) is one abstract class: a
  `LinearCategory`, `TracedCategory` and `SymmetricCategory`, with the colaxator, the trace
  preserved by the bang, and a reflexive object `U` with `U @ U`, the unit and `!U` as
  retracts. The inherited linear laws stay the comonad and comonoid equalities. There is no
  free GoI situation, no `linear.feedback` and no `feedback.linear`: `discopy.linear` and
  `discopy.feedback` do not import each other, and `discopy.interaction` is where they meet.

## Consequences

- The names that moved have no alias (ADR 1): `symmetric.Trace` is `feedback.symmetric.Trace`,
  `markov.FollowedBy` is `feedback.markov.FollowedBy`, `symmetric.Diagram.wait` is
  `feedback.symmetric.Diagram.wait`, and so on; `closed.Timed` is `feedback.closed.Timed`.
- A plain `markov` or `closed` diagram reads `str` and `repr` with `monoidal.Ty`, not
  `feedback.Ty`; closed terms, i.e. lambda terms, live in plain `closed` and do not delay.
  `closed.Diagram.to_compact` is `feedback.closed.Diagram.to_compact`, since it needs a trace.
- `egraph.Join` is a Markov bubble and no longer delays; `neural.Network` is traced through
  `feedback.markov` and `feedback.comarkov`.
- `discopy.bang` is gone with no alias: `bang.X` is `linear.monoidal.X`. A Markov, closed or
  compact level of `linear` waits on a later ruling.
- `Hypergraph.plug`, which rewriting calls, glues the hole into its context with
  `Hypergraph.glue`, not `trace`: the pushout of a convex match needs no trace in the category,
  so plain symmetric hypergraphs still rewrite.

## Status

Accepted, 2026-11-25. [#737](https://github.com/rel-int/discopy-next/pull/737), which carries
it out, is open.
