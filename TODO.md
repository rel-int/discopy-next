> Review round on #693 (cubic-dev-ai[bot]), head 0dd5a511:
>
> 1. discopy/optyx/core/path.py:626 (thread 4656189888)
> [P2] `Sum.prob` can truncate probabilities for mixed-dtype sums because it uses the first term's dtype instead of the promoted dtype returned by `eval`; construct the result using `amplitudes.dtype` so values such as `0.5` remain probabilities.
>
> ```suggestion
>         return Probabilities[amplitudes.dtype](
> ```
>
> 2. discopy/optyx/core/path.py:582 (thread 4656189892)
> [P2] Typed empty sums compare equal even though they produce different amplitude dtypes. Include `self.dtype` in the setoid so equality reflects the declared coefficient ring.
>
> ```suggestion
>         return (factory_key(type(self)), self.terms, self.dom, self.cod, self.dtype)
> ```
>
> 3. docs/adr/0011-path-matrices-add-formally.md:40 (thread 4656189897)
> [P3] Include the carrying PR in the status while this change is unmerged, as required by the ADR index. This preserves traceability from the decision to its implementation.
>
> ```suggestion
> Accepted, 2026-11-22. #693, which carries it out, is open.
> ```

- [ ] path.py:626 Sum.prob builds Probabilities from the dtype eval promoted to, not the first term's
- [ ] path.py:582 the dtype is part of a sum's setoid, and of its repr so it reads back equal
- [ ] the ADR status names #693 while it is open
