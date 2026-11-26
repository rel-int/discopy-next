# 13. Closed is a package

## Context

`discopy.closed` was the free closed Markov category: exponentials and copy and discard in one
module, `abc.ClosedCategory` a `BiclosedCategory` and a `MarkovCategory`. `discopy.biclosed` was
the planar level beside it. A compact category is symmetric monoidal closed, with
`x ⊸ y = x* @ y`, and the free one has no copy, so `compact` could not subclass `closed`.
Alexis ruled on 2026-11-25: "compact is not a subclass of closed which is a mistake due to fear of
boilerplate, we should fix it. that means we can't have closed be markov by default, we could
have a whole subpackage with closed.monoidal (current biclosed), closed.symmetric and
closed.markov", and "let's do the closed split in a first PR", then: "it seems like we need
a subsubpackage feedback.closed.symmetric and feedback.closed.markov".

## Decision

- `discopy.closed` is a package of three modules. `closed.monoidal` is the free biclosed
  category, with `x << y` and `y >> x` and planar linear terms. `closed.symmetric` is the free
  symmetric closed category, where the two exponentials are one, evaluated and curried on
  either side. `closed.markov` adds copy and discard, with the lambda terms, tuples,
  projections, `let` and Böhm trees.
- `abc.ClosedCategory` is a `BiclosedCategory` and a `SymmetricCategory`.
  `abc.ClosedMarkovCategory` is a `ClosedCategory` and a `MarkovCategory`: what
  `abc.ClosedCategory` was.
- `abc.CompactCategory` is a `ClosedCategory` and `compact.Diagram` a `closed.symmetric.Diagram`.
  Its evaluation and currying are the cups and caps of `rigid`, its trace those of `pivotal`.
- `discopy.feedback.closed` is a package too: `feedback.closed.markov` is feedback over
  `closed.markov`. `closed` does not import `feedback`.

## Consequences

- The old names keep no alias (ADR 1): `biclosed.X` is `closed.monoidal.X`, `closed.X` is
  `closed.markov.X` and `feedback.closed.X` is `feedback.closed.markov.X`. A tree or a pickle saved under them does not load.
- A compact diagram has left and right exponentials `y @ x.l` and `x.r @ y`, which are not
  equal, so `currying_symmetry`, which takes them to be one, is inapplicable to every
  `abc.CompactCategory`.
- For C3 to order the classes of `compact.Diagram`, `closed.symmetric.Diagram` lists
  `closed.monoidal.Diagram` before `symmetric.Diagram`, and `BalancedCategory` after
  `ClosedCategory`, as `ribbon.Diagram` has them. `compact.Diagram` lists `ribbon.Diagram`
  first and keeps the `braid_factory`, `twist_factory`, `term_factory` and `dagger_monoidality`
  of `symmetric.Diagram` by name, as `frobenius.Diagram` keeps the `map_agreement` and
  `term_factory` of `markov.Diagram`: every class below them resolves each name as before.
- A `closed.symmetric` type makes no terms yet: the linear lambda calculus with one exponential
  is a later pull request, and `closed.markov` has the terms it had.
- `rigid` and `pivotal` were under `closed.monoidal` already: `rigid.Diagram` subclasses it.
- `feedback.closed.symmetric`, the traced symmetric closed category with no copy, is the next
  pull request.

## Status

Accepted, 2026-11-25. [#743](https://github.com/rel-int/discopy-next/pull/743), which carries it
out, is open.
