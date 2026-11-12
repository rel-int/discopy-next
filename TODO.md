> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [x] Slice 2 of 3, third part, after #527. `union` keeps the invariant of the builder, relates its two spiders and no more, and lowers what `rebuild` has left to do
- [x] `rebuild` empties the worklist with its fuel, keeps the invariant and relates no more than a congruent equivalence containing the classes
- [ ] A builder with the invariant freezes into a valid e-graph, and `merge` and `rebuild` on it are `merge` and `rebuild` on that e-graph
- [ ] Docs: `lean/README.md`, a CHANGELOG.md `Project` bullet
