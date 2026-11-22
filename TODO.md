> Review round on #693 (cubic-dev-ai[bot]), head b4e75601:
>
> 1. discopy/optyx/core/path.py:612 (thread 4656265491)
> [P1] `Sum.eval` can silently add amplitudes from different photon-number sectors. `Amplitudes` stores only occupation-basis lengths, so one-mode sectors all have shape `1x1`; compare each term’s output photon count before adding and raise `AxiomError` when they differ.
>
> 2. discopy/optyx/core/path.py:601 (thread 4656265500)
> [P2] `Sum.tensor` drops the dtype whenever the distributed product is empty. Tensoring formal zeros therefore changes `Matrix[float]` into a complex sum and changes the dtype of `eval`; pass the appropriate operand/promoted dtype when constructing an empty result.
>
> 3. docs/adr/0011-path-matrices-add-formally.md:28 (thread 4656265507)
> [P2] This API guidance is misleading: `eval` and `prob` select a sector but return objects without `.creations`, so callers still get `AttributeError` when they need the creation profile. Direct them to inspect `path.Sum.terms` for profiles and use `eval` or `prob` only for sector results.

- [x] path.py:612 Sum.eval refuses terms that send a sector to different sectors, which one mode out hid
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:52 path.py:601 the ring survives every operation of cat.Sum, not just the ones that leave a term
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:52 the ADR says where the creation profiles are: on the terms, not on what eval returns
