# A box is mapped by its method: bubbles fuse into boxes, each level gets its plumbing

## Context

Functors map structural boxes through dozens of `isinstance` branches, each level adding its own.
This has caused regressions again and again, and it blocks constructions acting on functors
([#33](https://github.com/rel-int/discopy-next/issues/33), decision 3 of
[#56](https://github.com/rel-int/discopy-next/issues/56)). toumix ruled live on three changes
toward one rule:

- [#513](https://github.com/rel-int/discopy-next/issues/513), 2026-11-12: a symmetric functor maps
  a permutation by one call on its codomain.
- [#560](https://github.com/rel-int/discopy-next/issues/560), 2026-11-16: "Bubble disappears by
  fusing with the Box class, ie a Box is a Bubble with zero arguments".
- [#561](https://github.com/rel-int/discopy-next/issues/561), 2026-11-16: braided, pivotal,
  ribbon and compact get their plumbing, `BraidWord`, `TemperleyLieb`, `Tangle` and `Brauer`,
  beside symmetric's `Permutation`.

## Decision

A box is an operation symbol of arity n. It has `args`, n arrows, and a generator is a box with
none. A functor sends a box with no args through `ar_map`, and any other box by one call
`getattr(cod, method)(*map(self, args), dom=..., cod=...)`. The plumbing of a level, the part of
a diagram fixed by coherence, is one value per layer, kept in normal form. A functor maps it by
one call on its codomain, by level, as #513 does for a permutation.

## Consequences

- `Bubble` and `bubble_factory` go from every module, and `isinstance(f, Bubble)` becomes
  `f.args`.
- The plumbing classes form the same diamond as the categories: ribbon's joins braided's and
  pivotal's, and compact's absorbs ribbon's and `Permutation`.
- Each slice goes upstream separately, when toumix asks.

## Status

Accepted, 2026-11-16. #560 is carried out by
[#729](https://github.com/rel-int/discopy-next/pull/729), with toumix's rulings of 2026-11-25:
`Bubble` stays importable as an alias of `Box.from_args` (of `Elementwise.from_args` in `tensor` and
`quantum`), `Sum` does not fuse yet, and the function
a tensor bubble applies moves to `tensor.Elementwise`, a tensor box with args. #513 and #561 are
open.
