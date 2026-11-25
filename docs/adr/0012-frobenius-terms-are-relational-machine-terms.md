# 12. The terms of a Frobenius diagram are those of the relational machine calculus

## Context

[#658](https://github.com/rel-int/discopy-next/pull/658) gave each module below `biclosed` a kappa
calculus, and [ADR 8](0008-a-type-of-its-level.md) gave each level a type that knows its terms.
Above `biclosed` a type applied to a function is a lambda abstraction, and a Frobenius diagram read
as the symmetric kappa term of `markov`, which needs exactly one output, so a cup, a cap or an
effect had no term at all.

The relational machine calculus of Barrett, Castle and Heijltjes (LICS 2024,
[arXiv:2405.10801](https://arxiv.org/abs/2405.10801)) splits kappa abstraction into a binder and a
pop, `κx. M = ∃x. ⟨x⟩ ; M`. Its terms have a syntactic dagger: reverse the term and exchange push
and pop. Its typed fragment with no sums, no star and no function symbols is the free extra
hypergraph category (Theorem 6.5). toumix and Joseph Fourment asked for it on 2026-11-24 and agreed
its design and syntax in [the thread that built it](https://ampcode.com/threads/T-aa6d903b-73cd-4bd5-a7a3-2b9f8e77bf39).

## Decision

`discopy.frobenius` has a `TermBase` whose classes are the constructs of the calculus, one Python
form each:

- a value used as a term is its push, either a `Variable` or a box applied to values, an
  `Application`; `t.at(a)` pushes it on location `a`, a `Push`
- `t.dagger()` is its pop, a `Pop`
- `M >> N` is a `Sequence`, and `M + N` and `M.star()` are a `Choice` and a `Star`
- `A.exists(lambda x: M)` is an `Exists`

`A(lambda x: e)` is kappa abstraction at every level, here `∃x. ⟨x⟩ ; e`. So a term written at a
lower level reads at this one as sugar, and the binders the lower levels lack get methods of their
own.

A term evaluates by running the machine on spiders rather than values: a pop merges spiders and a
box applies to the head of the stack. What the machine builds is the `Hypergraph` of the term, and
a hypergraph reads back as a term in normal form. The pop of an `Application` applies the box
backwards, its `Diagram.converse`, so the dagger of a term evaluates to the converse of its diagram.

Locations are there from the start, so that other models compile to these terms. A location is a
`str`, the empty one the default stack. The domain of a term is one block per location, the default
first and then the others by name, with the head of each stack the last wire of its block.

## Consequences

- `frobenius.Ty('x')(lambda a: ...)` is no longer a `biclosed` lambda term, and
  `frobenius.Diagram.to_term()` gives a term for every diagram, of any number of outputs.
- A frobenius box applied to values is an `Application`, unless a class below it calls boxes
  otherwise, as `Diagram.from_callable` does while it runs.
- A variable bound and never used is a spider with no leg, which `frobenius.Equation` does not
  remove: DisCoPy's spiders are special but not extra, so the ν law `∃x. ⋆ = ⋆` fails.
- The laws of unification hold of values but not of the boxes they evaluate to, so a term with
  boxes evaluates soundly for β, η and ν only.
- A `Star` has no diagram, and a `Choice` evaluates to the formal sum of its branches.
- The levels from `compact` down to `symmetric` are restrictions of how a variable is pushed and
  popped. They are not built yet.

## Status

Accepted on 2026-11-25, carried out by the pull request that adds this record.
