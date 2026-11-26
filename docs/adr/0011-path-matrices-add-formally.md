# 11. Path matrices add formally

## Context

`optyx.core.path.Matrix` carries photon `creations` and `selections`, so its array has
`dom + len(creations)` rows and `cod + len(selections)` columns. It inherited
`matrix.Matrix.__add__`, which adds the arrays and rebuilds the class from `(array, dom, cod)`
alone: two matrices with different profiles had arrays of different shapes, two with the same
profile lost their `creations`, `selections`, `normalisation` and `scalar`, and
`cat.Functor.__call__` on a `Sum` added each term to a `zero` with no creations at all, which is
how `grad` through a composed `photonic.Phase` failed
([#544](https://github.com/rel-int/discopy-next/issues/544)). Three readings were put on #544 and
toumix's fleet picked the first on 2026-11-20;
[#693](https://github.com/rel-int/discopy-next/pull/693) carried it out.

## Decision

A sum of path matrices is formal: `path.Matrix.sum_factory` is `path.Sum`, a `cat.Sum` of
matrices with the same number of modes in and out, `path.Matrix.zero` is the empty sum, and
`then`, `tensor` and `dagger` distribute over it. The sum is evaluated, not collapsed:
`Sum.eval(n_photons)` adds the terms' `Amplitudes`, matrices over the occupation numbers of
`n_photons` photons, which add however the terms create and select, and the empty sum evaluates
to zero.

## Consequences

- `to_path` of a sum of diagrams is a `path.Sum`, not a `path.Matrix`. A caller after a creation
  profile reads it off `Sum.terms`, one `Matrix` each; `eval` and `prob` answer for a photon
  sector, and what they return — `Amplitudes` and `Probabilities` — has no profile of its own
  either, being indexed by occupation numbers rather than by modes.
- Addition no longer claims that summing the matrices sums the maps, which only holds on the
  one-photon sector: `(Endo(2) + Endo(3j)).to_path().eval(2)` is `2² + (3j)²`, where collapsing
  to `Endo(2 + 3j)` gave `(2 + 3j)²`.
- Two of the readings on #544 are ruled out: addition restricted to matching profiles, which
  says nothing about the general sum, and `to_path` refusing a `Sum`, which leaves `grad`
  unusable through `to_path`.
- A sum whose terms change the photon number by different amounts sends one sector to several, and
  `eval` raises `AxiomError` on it rather than adding amplitudes indexed by different occupation
  numbers: the shapes alone do not catch it, since one mode out has one element per sector.
- A sum answers for one ring: the `dtype` of its terms, or, where there is no term, the one the
  `Matrix` it came from named, carried through every operation and part of both its equality and
  its representation.

## Status

Accepted, 2026-11-22. [#693](https://github.com/rel-int/discopy-next/pull/693), which carries
it out, is open.
