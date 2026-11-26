# 14. Traced levels with duals

## Context

After [ADR 13](0013-closed-is-a-package.md), `pivotal`, `ribbon`, `compact` and `frobenius`
were rigid categories whose trace was a class method composing a cap, the diagram and a cup,
beside a `feedback` package whose levels carry a delay and a trace box, and a `closed` package
whose levels carry exponentials. Alexis settled the shape of the hierarchy on 2026-11-26: "I
think we finally got round to a convincing hierarchy, write it that way", with
"ha yes ofc rigid sits under closed.monoidal which replaces the current biclosed, never liked
that word anyway" and "rename it in the abc too, ClosedMonoidalCategory vs
ClosedSymmetricCategory". His analogy for the delay of a level with duals: "every symmetric
category is balanced (with the trivial twist)".

## Decision

- `abc.BiclosedCategory` is `abc.ClosedMonoidalCategory` and `abc.ClosedCategory` is
  `abc.ClosedSymmetricCategory`.
- `feedback.closed` has `monoidal`, `balanced`, `symmetric` and `markov`, each the free feedback
  category over the closed level of its name, `balanced` over `closed.monoidal` with the braids
  and twists of `feedback.balanced`: there is no `closed.balanced`.
- The plain levels without duals are not traced. Each level with duals sits under its
  `feedback.closed` level, free modules included: `pivotal` under `feedback.closed.monoidal`,
  `ribbon` under `feedback.closed.balanced`, `compact` under `feedback.closed.symmetric`,
  `frobenius` under `feedback.closed.markov` and `compact`.
- On them the delay is the identity by default, as `CompactCategory.twist` is, so
  `Feedback(f)` is `Trace(f)`, and a wire or a box built with a time step raises rather than
  being reset in silence. `Trace` stays a box of its own, as `Spider(0, 2)` stays distinct
  from `Cup`, and `PivotalCategory.trace_duals` equates it with a cap, `f` and a cup. A semantic
  category may choose another delay, a strong monoidal endofunctor, which preserves duals; in a
  `HypergraphCategory` it preserves the spiders too, `spider_delay`.
- `Head`, `Tail` and `FollowedBy` are carried down; a functor into a codomain without streams
  raises on them, as it did on a trace.
- The planar level has no Int construction. It starts at `balanced`, and so does the theorem of
  Hasegawa that a traced category is closed if and only if its embedding into Int has a right
  adjoint.
- `linear.symmetric` takes the swap for the exchange, `linear.markov` the copy and discard of a
  banged object for its contraction and weakening, with no default bang: the identity is a
  linear exponential comonad only when every morphism is deterministic.
- `linear` and `closed` import nothing from `feedback`, `feedback` nothing from `linear`.

## Consequences

- `pivotal.Diagram.trace_factory` is the `Trace` box rather than a class method. A category
  whose trace is not a box, e.g. `tensor.Diagram`, `quantum.circuit.Circuit`, the diagrams of
  optyx and `peirce.beta`, sets `trace_factory = dual_trace` and keeps a time-stepped delay of
  its own. `Hypergraph.explicit_trace` and `CMap.explicit_trace` open a pivotal trace into its
  cups and caps, as they did.
- A drawing of a trace at these levels labels its loop with the traced type, not its adjoint:
  eight baselines of `docs/_static/int` and `docs/_static/ribbon` changed by that label, and
  `int/simplify.svg` also draws its nested traces as nested loops.
- `compact.Swap` is not a `ribbon.Braid`, since the two `Box` chains cannot both be ordered by
  C3; `ribbon.DualRail` dispatches on `balanced.Braid` to keep sending it to dual rail braids.
- The rotation and adjunction of pivotal diagrams hold, the feedback there being a trace.
- No `letrec`: a cycle in a term is the push-before-pop of the relational machine calculus,
  [#734](https://github.com/rel-int/discopy-next/pull/734).

## Status

Accepted, 2026-11-26. [#744](https://github.com/rel-int/discopy-next/pull/744), which carries it
out, is open.
