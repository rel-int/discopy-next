> toumix on toumix/memory#202 (https://github.com/toumix/memory/issues/202#issuecomment-6765151788):
>
> > move the PR to next and incorporate the comments I left there then close the upstream PR
>
> toumix's review comments on discopy/discopy#533:
>
> > can we use the same name for the attribute and the parameter?
>
> > let's replace the bare string with an enum
>
> > I find it too confusing to have three different words `contract`, `contractor`, `contraction` for almost the same thing, we need to find a better design. In fact, even `backend` is doing the same job already, can we just have a `Contractor` subclass of `Backend`?

- [x] `matrix.Contractor(Backend)`: a backend whose `einsum` contracts the arrays of another backend, with `OptEinsum` and `Quimb`; a plain `Backend` is the einsum engine
- [x] one word: `contractor` is the parameter and the attribute of `tensor.Functor`, and the parameter of `Diagram.eval` and `Circuit.eval`
- [x] keep #533's `Functor.operands` and `Functor.to_quimb`, its tests adapted, its `tensor` extra and README line
- [x] CHANGELOG bullets and the checks of CONTRIBUTING.md
