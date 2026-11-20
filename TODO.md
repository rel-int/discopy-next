> 🌙 Evening, 2026-11-20. Autonomous round: the next repo's open bugs that need no
> decision, oldest first (DESIRE_REPO `EVENING.md` step 5).
>
> rel-int/discopy-next#652, `toumix-agents`, 2026-11-19 15:51Z, verbatim:
>
> > **Matrix of a list of PyTorch scalars drops their gradient**
> >
> > Found while fixing #650 (optyx gates in the backend of their parameters).
> >
> > Under `matrix.backend("pytorch")`, `Matrix([t], 1, 1)` with `t` a 0-d tensor that
> > requires grad builds its array with `torch.as_tensor([t])`, which detaches `t`
> > (PyTorch warns "Converting a tensor with requires_grad=True to a scalar"). So a
> > photonic gate's array now keeps its gradient (#650), but `gate.to_path().array`
> > does not: `zw.Endo(scalar).to_path()` is `Matrix[dtype]([self.scalar], 1, 1)`, and
> > `zw.Endo.__init__` also calls `complex(scalar)`, which detaches too.
> >
> > ```python
> > import torch
> > from discopy.matrix import backend
> > from discopy.optyx.photonic import MZI
> > t = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
> > with backend("pytorch"):
> >     assert MZI(t, 0.2).to_path().array.grad_fn is None  # should not be
> > ```
> >
> > JAX is fine: `jnp.asarray([tracer])` keeps the trace, and #650's tests go through
> > `to_path()` under JAX.
> >
> > Fix: stack a list of backend scalars (`torch.stack`) in `Matrix.__init__` when the
> > entries are tensors, and skip `complex()` in `zw.Endo` for a non-numpy scalar.
> > Touches `matrix.py`, a core module, so not folded into #650.

- [ ] Reproduce on main: Matrix([t], 1, 1) detaches, and so do MZI(t, .2).to_path().array and zw.Endo(t).to_path().array
- [ ] matrix.asarray: the array of nested entries, stacking the ones the backend already owns
- [ ] Matrix.__new__ and Matrix.__init__ build their array with it, and photonic.Gate.asarray delegates rather than repeating the stack
- [ ] zw.Endo keeps a scalar its backend owns, and conjugates it in that backend rather than through complex
- [ ] Tests: the issue's assertion under pytorch, the same under JAX, and the numpy path unchanged
- [ ] CHANGELOG entry, then every gate of CONTRIBUTING.md's Run the unit tests
