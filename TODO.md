> cubic-dev-ai review of #784 at 5c92da46, 10 issues: multi-wire and unit `Wait` drawing (backend.py:1499); no Lean typing rule for `wait` (abc.py:684); daggered `Wait` into a codomain with `wait` but no dagger (feedback/cat.py:301, feedback/__init__.py:1468); `Wait` missing from the autosummaries of closed/symmetric, ribbon, compact, closed/balanced; feedback/symmetric docstring states the free equality as fact; changelog wording.

- [x] backend.py: draw a loop on every wire of a multi-wire Wait, none on the unit
- [x] abc.py: a waiting rule typing wait(x): x -> x.d, ABC.lean regenerated
- [ ] feedback and feedback.cat functors: a daggered Wait goes through wait only into a dagger codomain
- [ ] autosummaries: Wait in every module that exports it
- [ ] feedback/symmetric docstring and changelog wording; Stream is not enrolled
- [ ] codex: tensor.Diagram.wait builds a frobenius.Wait, not a tensor diagram
- [ ] codex: neural Unroll expands a Wait as the swap fed back, so rounds=1 keeps its zero memory
