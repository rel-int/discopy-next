> With this egraph project on discopy-next I'm starting to feel the need to write formal proofs of correctness beyond the property testing, help me brainstorm on how to use Lean to formalise discopy

> ok drop the python interpreter formalisation for now, spawn a fresh session to go through option A

- [WIP] @session_01MEJxy2cXv66NHgL7vwn3q8-2026-11-12 19:04 Slice 2 of 3, second part, after #525. The invariant of the builder, stated on the roots: forest, least, keys, hash-cons, parents, worklist, and the outputs of a node related to those of the node of its key
- [ ] `Hashed.cons` and `Tables.place`: entering a node in the hash-cons and placing a node keep it
- [ ] `Tables.link_plain`, `Moving.rekeys`, `Tables.linked`: a class joining another keeps it, the quick way and by moving its parents
- [ ] Docs: `lean/README.md`, a CHANGELOG.md `Project` bullet
