> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [x] Slice 2 of 3, last part, after #528. `add` on the builder is `add` on the e-graph it freezes into, with the same outputs, and the builder of an e-graph has the invariant
- [x] The theorem `building`, `Builder.build g operations = g.fold operations` for the histories the Python accepts, and `adding`, restated in `Statements.lean` with what each rules out
- [WIP] @session_01MEJxy2cXv66NHgL7vwn3q8-2026-11-12 19:42 Docs: `lean/README.md`, CONTRIBUTING.md, AGENTS.md, a CHANGELOG.md `Project` bullet
