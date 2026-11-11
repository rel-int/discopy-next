> > i also want to use his experiment as a test to implement a fast version of discopy's egraph/ehypergraph
>
> > yes add the extra test for cartesian, actually "ehypergraph" is an annoying name let's just rename it to "egraph" so that a ehypergraph is synonymous with a "monoidal egraph"
>
> > let's go with fast egraphs! https://github.com/rel-int/discopy-next/issues/499
>
> (toumix, 2026-11-11: the intent of #499 and its go. This branch carries slice 0 of that issue, the rename and the cartesian doctest.)

- [x] rename the module `discopy.ehypergraph` to `discopy.egraph` and the class `EHypergraph` to `EGraph`, `Markov = EGraph[markov.Diagram]` kept, no alias, in the 10 files that name them, with `git mv` so history follows
- [x] the module docstring opens with the definition: an e-hypergraph is a monoidal e-graph
- [x] a doctest of the cartesian case: egg's e-graph is an `EGraph` fed cartesian boxes with one output each, `rebuild` merges two equal cartesian boxes and never two samples
- [ ] the proptest report loses no cell against `main` except renamed ones, each named in the pull request
- [ ] one `Changed` bullet in CHANGELOG.md
