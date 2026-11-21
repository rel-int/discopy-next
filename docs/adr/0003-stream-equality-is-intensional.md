# 3. Stream equality is intensional

## Context

A `stream.Ty` or `stream.Stream` is its first step `now` and a thunk `_later` for the rest.
Equality compares the thunk by identity, so two non-constant streams that unfold to the same steps
compare unequal ([#457](https://github.com/rel-int/discopy-next/issues/457)). Equality that unfolds
the stream would not terminate on infinite streams, and a bounded unfolding is a different
relation. toumix picked option 4 on #457 on 2026-11-09, and
[#535](https://github.com/rel-int/discopy-next/pull/535) carried it out.

## Decision

`==` on streams stays intensional: identity on a non-constant stream, structural on a constant
one, and the docstrings of `stream.Ty` and `Stream` say so. A law of streams is read with
`stream.up_to_steps`, the equation of the underlying category on the first `stream.STEPS` steps,
not with `==`.

## Consequences

- `==` on streams says nothing about a law, and a test that compares two streams with `==` tests
  identity.
- `Stream.equation_factory` and `abc.Category.ob_equation` carry the relation a law uses, apart
  from `==`.
- The property matrix does not check streams until
  [#533](https://github.com/rel-int/discopy-next/issues/533) is fixed. That half of #457 stays open.

## Status

Accepted, 2026-11-13.
