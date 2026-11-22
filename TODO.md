> ask, one 🚀: formal sums in the codomain, reading 1 above. `path.Matrix` gets a `sum_factory`, `zero` is the empty sum, and `eval(n)` adds its terms' amplitudes on the n-photon sector. tests: the three cases of the report and the `4j + 10j == 14j` cell.
>
> not taken: `+` only where creations and selections match (2), `to_path` refusing a `Sum` (3).
>
> default: reading 1. asked 2026-11-20, due 11-22: with no word an agent builds it then, and one word reverses it.

- [x] path.Sum, the formal sum of path matrices: reuse cat.Sum, with eval(n) adding its terms' Amplitudes on the n-photon sector
- [x] path.Matrix.zero is the empty sum and + builds one; then and tensor go through it
- [ ] test the three cases of the report, the 4j + 10j == 14j cell and the repr roundtrip
- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:35 measure #252 (doubling a Sum evaluating to zero) against this change and report
- [ ] CHANGELOG bullet with the PR number, and the gates of next-checks
