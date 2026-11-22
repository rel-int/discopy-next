> Review round on #693 (cubic-dev-ai[bot]), head dc1b8e3a:
>
> 1. discopy/optyx/core/path.py:595 (thread 4656112469)
> [P2] `Sum.prob` drops `Matrix.prob`’s `with_perceval` argument. Diagram sums now return this class, so existing `prob(..., with_perceval=True)` calls fail with `TypeError`; preserve the interface or reject it explicitly.
>
> 2. discopy/optyx/core/path.py:570 (thread 4656112475)
> [P3] `Sum.setoid()` makes non-empty path sums unhashable through the inherited `Arrow.__hash__`. This breaks callers that use formal sums as categorical keys or set members; provide a hash consistent with matrix terms, or explicitly make the sum unhashable instead of inheriting a failing hash implementation.
>
> 3. test/optyx/test_diagram.py:53 (thread 4656112477)
> [P3] This assertion compares `terms.to_path()` with a newly constructed copy of the same diagram sum, so it cannot detect an incorrect `to_path()` representation. Compare against the explicit sum of the two term path matrices instead.
>
> ```suggestion
>     assert terms.to_path() == (
>         zw.Endo(2).to_path() + zw.Endo(3j).to_path())
> ```
>
> 4. discopy/optyx/channel.py:1461 (thread 4656112487)
> [P3] This reference links to `channel.Sum`, but `to_path` returns a formal `path.Sum` of matrices. Qualify the reference so generated documentation points to the returned type.
>
> 5. test/optyx/test_path.py:100 (thread 4656112490)
> [P3] This assertion does not verify the empty sum's tensor shape because `np.allclose` broadcasts scalar `0`; a scalar result would pass. Compare against an explicitly shaped zero tensor so regressions in the promised shape are caught.
>
> 6. discopy/optyx/core/path.py:562 (thread 4656112493)
> [P3] `Sum` hardcodes `dtype=complex`, so empty formal zeros lose the dtype of `Matrix[float]` or another specialized path matrix. Carry the factory’s dtype into the formal sum so zero evaluation matches nonempty paths.
>
> 7. docs/adr/0011-path-matrices-add-formally.md:1 (thread 4656117136)
> [P2] This ADR is numbered 8 in its heading but 0011 in its filename, so the documented decision conflicts with the repository’s ADR numbering and can be confused with another record. Change the heading to `# 11.`.
>
> ```suggestion
> # 11. Path matrices add formally
> ```

- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:10 path.py:595 Sum.prob keeps Matrix.prob's with_perceval position and rejects it with a reason
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:10 path.py:570 say the sum is unhashable rather than inheriting a hash that raises
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:11 test_diagram.py:53 compare to_path() against the sum of the terms' path matrices
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:11 channel.py:1461 qualify the Sum reference as path.Sum
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:11 test_path.py:100 check the empty sum's tensor shape, not a broadcast scalar
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 02:10 path.py:562 carry the path matrix's dtype into its formal sums
- [ ] the ADR heading reads 11, already fixed in 434097c9
