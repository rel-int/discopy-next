# Equality counts `is_dagger`

## Context

A functor sends the dagger of a generator to the dagger of its image, so it is well defined only if
the equality of generators separates whatever the functor can separate. Two equalities did not.
`monoidal.Wire` compared name and colours but not `is_dagger`, so `Ty('x')[::-1] == Ty('x')`
while their images differed. `cat.Bubble.setoid` also left out `is_dagger`, though
`tensor.Bubble` counted it, so equal bubbles had unequal tensor images. toumix ruled on both: the
wire in [#464](https://github.com/rel-int/discopy-next/pull/464), which carried his calls on the
review [#445](https://github.com/rel-int/discopy-next/issues/445), and the bubble in
[#507](https://github.com/rel-int/discopy-next/pull/507).

## Decision

The dagger of a wire and the dagger of a bubble are generators of their own: equality and hashing
count `is_dagger`, and `str` prints it as `[::-1]`. A bubble does not commute with dagger.

## Consequences

- Next diverges from upstream on purpose: upstream states that a bubble commutes with dagger
  ([discopy#629](https://github.com/discopy/discopy/pull/629)) and its test asserts it. Next's
  `test_Bubble_dagger` asserts the opposite.
- A reversed coloured wire no longer equals a fresh wire with its colours swapped.
- A box that is its own dagger says so with `is_dagger=None`, as `quantum.gates.SWAP` and optyx's
  `H` ([#498](https://github.com/rel-int/discopy-next/pull/498)) do, not by being equal to it.

## Status

Accepted, 2026-11-12.
