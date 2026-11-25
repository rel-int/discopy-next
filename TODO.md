> Review round on #737, cubic-dev-ai at 71d3a0a:
>
> > [P2] `glue` skips the boundary checks that `trace` performs, so invalid counts can silently glue fewer wires and mismatched types can be merged. Call `assert_istraceable(self, n, left)` before the zero-count case.
> > [P3] The `GoISituation` entry omits the symmetric requirement. `abc.GoISituation` is linear, traced and symmetric, so name it as a symmetric traced bang.

- [x] discopy/hypergraph.py glue checks its boundary as trace does
- [x] CHANGELOG.md GoISituation is a symmetric traced bang
