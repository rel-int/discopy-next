# Stream equality is intensional

## Context

A `stream.Ty` or `stream.Stream` is its first step `now` and a thunk `_later` for the rest.
Equality compares the thunk as a Python object, which for a lambda is identity, so two
non-constant streams that unfold to the same steps compare unequal ([#457](https://github.com/rel-int/discopy-next/issues/457)). Equality that unfolds
the stream would not terminate on infinite streams, and a bounded unfolding is a different
relation. toumix picked option 4 on #457 on 2026-11-09, and
[#535](https://github.com/rel-int/discopy-next/pull/535) carried it out.

## Decision

`==` on streams stays intensional: it compares the representation, the thunk included, so it is
identity on a non-constant stream built from a lambda, and the docstrings of `stream.Ty` and `Stream` say so. A law of streams is read with
`stream.up_to_steps`, the equation of the underlying category on the first `stream.STEPS` steps,
not with `==`.

## Consequences

- `==` on streams says nothing about a law, and a test that compares two streams with `==` tests
  identity.
- `Stream.equation_factory` and `Stream.ob_equation` carry the relation a law uses on arrows and on
  types, apart from `==`.
- The property matrix checks streams with that relation, and the three feedback laws up to a
  permutation of the memory, `stream.up_to_memory`
  ([#533](https://github.com/rel-int/discopy-next/issues/533)).

## Status

Accepted, 2026-11-13.
