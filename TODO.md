> i also want to use his experiment as a test to implement a fast version of discopy's egraph/ehypergraph
>
> let's go with fast egraphs! https://github.com/rel-int/discopy-next/issues/499
>
> (toumix, 2026-11-11: the intent of #499 and its go. This branch carries slice 1 of that issue: `Builder`, the law that it equals the pure fold, and the correctness replay of three traces.)

- [x] `egraph.Builder`, a mutable e-graph with a union-find, a hash-cons of deterministic boxes, the parents of each class and a worklist: `add`, `merge`, `rebuild`, `find`, `freeze`; `EGraph.builder` starts one from a value
- [x] the law `EGraph.building`: a builder freezes into the pure fold of the same operations, over `History`, random operations whose merges make boxes congruent, in the proptest matrix
- [x] unit tests on hand-built operations, each failing on a builder with a part removed
- [WIP] @session_01MEJxy1eq4ERdiEMTUzTY22-2026-11-11 13:40 replay of 3 small traces of #500 in `test/egraph.py`: the frozen partition is the recorded one, and a check that fails if `Builder.rebuild` is a no-op
- [ ] CHANGELOG `Added`, docs summary, every check of CONTRIBUTING.md
