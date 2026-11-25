# 12. Constructors are formation rules

## Context

A category's morphisms are derived from its rules, e.g. `cut` and `mix`, which is how
`cat.Arrow.search` and the rule matrix generate them. Its objects were not: `Tensor`, `L`, `R`,
`Delay`, `Over`, `Under`, `Bang`, `Unit` and `Repeat` were expressions in the signatures of rules
and axioms only, built by their `form`. Generating objects therefore went through each
category's own `Ty` strategy, and a sampler that derives both, e.g. a Boltzmann sampler on the
sequent calculus of a category, had no declaration of the objects to read.

## Decision

A constructor is the introduction rule of the sort of objects. `Constructor` subclasses `Rule`;
`K.formation(...)` is the rule whose premises are the objects `K` takes, one `Var[C0, X]` each,
`Var[nat, N]` for a number, and whose conclusion is the object it builds, `Var[C0, K[...]]`,
with `form` as its implementation. Each abstract base class declares the formation rules of the
constructors its structure introduces, named `ob_<constructor>`, and they are inherited like any
rule. A rule concluding an object says so with `Rule.forms_object`; rules that derive morphisms
leave those out.

## Consequences

- `Testable.rules` holds both kinds of rule; `Testable.constructors` holds the formation rules.
- The rule matrix checks every formation rule of every enrolled category. A category whose
  objects lack a constructor declares its formation rule inapplicable, as `finset.Cospan` does
  for the adjoints of natural numbers.
- `discopy.lean` leaves formation rules out: they are the operations on `Ob` its structures
  already have.
- The equations between objects, e.g. `x.r.l == x`, stay axioms of their own.

## Status

Accepted, 2026-11-25, on the branch `claude/boltzmann-types`.
