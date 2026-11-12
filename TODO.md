> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [WIP] @session_01MEJxy2cXv66NHgL7vwn3q8-2026-11-12 16:57 Slice 1 of 3, after #521. The pure e-graph in `lean/`: nodes and classes, `merge`, `add`, `rebuild` with its termination, and normal lists of classes
- [ ] Theorems `merge_equivalence`, `add_hashcons`, `rebuild_congruent`, `rebuild_extensive`, `rebuild_least` and `rebuild_closure`, restated in `Statements.lean` with what each rules out
- [ ] The oracle folds a history and `lean/test_oracle.py` compares it with `History.fold()`
- [ ] Docs: `lean/README.md`, CONTRIBUTING.md, AGENTS.md, a CHANGELOG.md `Project` bullet
- [ ] 50,000 histories over three seeds agree locally and one seeded mutant (`rebuild` ignoring `det`) fails the test, both stated in the pull request
