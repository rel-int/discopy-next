# 8. A frobenius winding is read modulo one

## Context

A frobenius object is self-dual: `Wire.l` and `Wire.r` are the object the other way round, and
`windings = (0, )`. The winding number `z` inherited from `rigid.Wire` was kept as it was given,
so a hand-built `Wire('x', 1)` was a wire whose adjoints never brought it back to `z == 0`, and
`rigid.Functor.__call__`, which sends such a wire to `self(other.l).r`, recursed until the stack
ran out ([#638](https://github.com/rel-int/discopy-next/issues/638)). The report named two
readings, which disagree about what a frobenius wire is: either the constructor refuses a nonzero
`z`, or `z` is read modulo the windings. toumix-agents picked the second on #638 on 2026-11-20,
and [#694](https://github.com/rel-int/discopy-next/pull/694) carried it out.

## Decision

A winding number is read modulo the windings of its level, so on a frobenius object, which has
one, it is always zero: `Wire('x', 1) == Wire('x')`. A nonzero `z` on a self-dual wire is a
number with a normal form, not a caller's mistake, and the normalisation happens in
`frobenius.Wire.__init__`, where the object is built, rather than in the functor that trips on it.

## Consequences

- `frobenius.Wire` accepts every `z` it accepted before and keeps none of them: `z` is not a
  distinguishing part of a frobenius object, and a caller who means to count windings wants a
  `rigid` or `pivotal` object.
- `rigid.Functor.__call__` terminates on every frobenius wire, since no frobenius wire has a
  nonzero `z` to recurse on.
- Every level below frobenius has one winding too, so the same normalisation holds of
  `quantum.circuit.Wire`, `peirce.beta.Wire` and `optyx.channel.Ob`.

## Status

Accepted, 2026-11-22.
