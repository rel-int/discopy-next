# TODO

> Move all of the optyx work into discopy-next, keep the license, make sure you keep the attribution of commits, copy all the open issues and pull requests adding giodefelice-agents as assignee. I had planned for him to do it (prompt attached) but I got some spare tokens so I'll do it myself instead.

toumix, 2026-10-22, interactive. Ruled in the same session: library PRs are ported and merged on
discopy-next, experiment PRs copied as drafts (toumix/desire#163).

Copy of rel-int/optyx#44 onto discopy.optyx (#225)

- [x] merge the move branch `origin/claude/magical-cannon-g1t6ms`
- [ ] port to `discopy.optyx` on next `main`: blocked. `QuimbBackend.eval` calls
  `tensor.Diagram.eval(contract="quimb", ...)`, which discopy/discopy#533 adds and next `main`
  does not have. #533 is giodefelice-agents' (head `b76559ca`, +346/−98, `tensor.py`,
  `quantum/circuit.py`, a `tensor` extra), branched from upstream at `a94378cc` before next's
  `tensor.Functor.contract` moved on, so it needs a port of its own. When it lands, its
  `contract_quimb` must also contract a `TensorNetwork` that `contract_compressed` returns, as
  optyx#15's `process_term` did, or the compressed `fix` loses that fix.
- [ ] checks: pytest on `discopy/optyx` and `test/optyx` 1150 passed, 132 failed, every failure
  `contract` (or `contract`, `max_bond`) reaching `einsum`; 1282 passed with the move's
  `QuimbBackend`. pflake8 and pylint's unused-* list clean.
- [ ] a CHANGELOG entry, once `QuimbBackend` contracts through discopy
