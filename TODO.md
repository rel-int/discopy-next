> cubic-dev-ai review https://github.com/rel-int/discopy-next/pull/691#pullrequestreview-6073884738:
>
> > test/lambek/modality.py:41 — This assertion does not verify that either closed reading has an empty domain. Assert `diagram.dom == Ty()` alongside the codomain.
>
> > docs/notebooks/strict_sloppy.md:145 — `amplitudes` silently discards the imaginary part of every output amplitude.
>
> > docs/notebooks/strict_sloppy.md:93 — `gate` crashes for a valid all-zero state with no inputs: `tuples` is empty, so this call evaluates `Wn(0)`, which recurses forever.

- [WIP] @T-01a1fa54-2026-11-26 12:10 assert the closed readings have an empty domain
- [WIP] @T-01a1fa54-2026-11-26 12:10 make amplitudes refuse an imaginary part instead of dropping it
- [WIP] @T-01a1fa54-2026-11-26 12:10 give Wn a base case with no outputs, the selection of the vacuum
