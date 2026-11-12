> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [x] Slice 2 of 3, first part, after #521 and #522. The `Builder` in `lean/`, method by method: `root` with path halving, `find`, `cons`, `place`, `add`, `union`, `merge`, `rebuild`, `freeze`
- [x] Path halving ends and keeps the root of every spider (`Forest.root_spec`)
- [x] The oracle builds a history, and `lean/test_oracle.py` compares it with `History.build()` on generated histories and on the 50 traces of `benchmark/data/egraph-traces`, plain and doubled then rebuilt
- [WIP] @session_01MEJxy2cXv66NHgL7vwn3q8-2026-11-12 17:45 Docs: `lean/README.md`, CONTRIBUTING.md, AGENTS.md, a CHANGELOG.md `Project` bullet
- [ ] 50,000 histories over three seeds agree locally and one seeded mutant of the builder fails the test, both stated in the pull request
