# 8. Each level has its types, and a type knows its terms

## Context

[#658](https://github.com/rel-int/discopy-next/pull/658) gave each module below `biclosed` the
terms of a kappa calculus, and made a type applied to a Python function the kappa abstraction of
that function. Several levels shared one class of types: `spacial` and `braided` used
`monoidal.Ty`, and `balanced`, `symmetric`, `markov` and `comarkov` used `feedback.Ty`. A type
could therefore not say which level its kappa was of, so the kappa was of the lowest level that
accepted the body. A closed term with no box, e.g. an identity, read back from its string as a
term of another class, and `str_transparency` failed on seven levels on `main`.

[#679](https://github.com/rel-int/discopy-next/pull/679) first weakened the law to an equation
between terms of any level. toumix ruled on it on 2026-11-21: "doesn't look very good, it would be
better to introduce a Ty subclass for each module with its own pointer to the Term class".

## Decision

Each module below `biclosed` has a `Ty` and a `Nat` of its own, the `ob` of its `Diagram`, which
subclass those of the levels its diagrams subclass. `Ty.term_factory` is the class of terms of
the module, and a type applied to a function is a kappa of that class. From `biclosed` up, where a
type applied to a function is a lambda abstraction, a `Ty` subclasses that of each level whose
boxes its diagrams take: `closed.Ty` is a `markov.Ty`, `frobenius.Ty` a `markov.Ty` and a
`comarkov.Ty`.

Types of two levels with the same class of wires are lists of the same wires. A type of another
level is read as one of the level it meets, by `Ty.cast_ty`: at the boundary of a box or of a
diagram, as the plumbing of a layer, and in a tensor of types, which is of the higher of the two
levels. Nothing else converts: types of two levels are not equal.

## Consequences

- `symmetric.Ty('x') != markov.Ty('x')`, and the repr of a type names its module.
- `str_transparency` is an equation between terms of one class again, at every level.
- Code that built a type in one module and compared it with the boundary of a box of another
  breaks. So does a type of lenses over Markov diagrams written `lens.Ty(x, y)`: it is
  `lens.Markov.ob(x, y)`.
- A tree or a pickle written with the shared types loads with those of its level.
- A level needs its `Nat` to be found from its `Ty`, hence `Ty.nat_factory`.
- `ribbon` and `compact` still share `pivotal.Ty`, which is neither a `balanced.Ty` nor a
  `symmetric.Ty` ([#683](https://github.com/rel-int/discopy-next/issues/683)).

## Status

Accepted, 2026-11-21, carried out by [#679](https://github.com/rel-int/discopy-next/pull/679).
