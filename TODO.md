> i also want to use his experiment as a test to implement a fast version of discopy's egraph/ehypergraph
>
> let's go with fast egraphs! https://github.com/rel-int/discopy-next/issues/499
>
> (toumix, 2026-11-11: the intent of #499 and its go. This branch carries slice 2 of that issue: the benchmark on the 50 traces, `dag_cost`, and what the profile of the builder asks for.)

- [WIP] @session_01MEJxy1eq4ERdiEMTUzTY22-2026-11-11 14:12 `benchmark/test_egraph.py`, suite `egraph`: the 50 traces through `Builder`, through the search's own bookkeeping on the same machine, through the pure fold on the small ones, and a rebuild of two copies
- [WIP] @session_01MEJxy1eq4ERdiEMTUzTY22-2026-11-11 14:09 profile the builder on the traces and fix what the profile shows, under 150 lines
- [ ] `EGraph.dag_cost` of a section and `dag_bound`, its greedy bound per class, with the law that sharing costs no more
- [ ] both targets of #499 measured, the table in the pull request, a miss filed as an issue
- [ ] CHANGELOG, CONTRIBUTING, every check of CONTRIBUTING.md
