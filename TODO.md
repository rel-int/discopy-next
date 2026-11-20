> > share as much as possible
>
> — toumix in the discopy-optyx Space, relayed by its thread: rework #674 so its terms share the existing term machinery (closed/biclosed) and optyx.Term instead of a new tensor.Term.

- [ ] drop tensor.Term: closed terms already copy with the codomain's copy
- [ ] fix #676 in closed.Abstraction.eval, with a regression test and the two closed baselines redrawn
- [ ] rewrite the notebook on closed terms and a closed.Functor into KExtension/BasisCopy
- [ ] run the gates and the notebook check
