> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [x] Slice 0 of 3. A Lake project in `lean/`, Lean 4.35.0 with no dependency: `partition` as a pure function and `partition_least`, the result is the canonical form of the equivalence the pairs generate
- [x] `lean/Statements.lean`, every theorem restated in one short file with the broken implementation it rules out, and `lean/Audit.lean`, failing on `sorry`, `native_decide` or an axiom beyond `propext`, `Quot.sound`, `Classical.choice`
- [x] The `oracle` executable, one JSON case per line, and `lean/test_oracle.py` comparing it with `egraph.partition`, skipped when the binary is not built
- [x] A `lean` workflow: build, audit and differential test on pull requests touching `lean/**` or `discopy/egraph.py` and on `main`
- [x] `lean/README.md` with the commit the model mirrors, CONTRIBUTING.md's "Run the Lean proofs", a CHANGELOG.md `Project` bullet
- [WIP] @session_01MEJxy2cXv66NHgL7vwn3q8-2026-11-12 16:27 50,000 cases over three seeds agree locally and one seeded mutant of the model fails the test, both stated in the pull request
