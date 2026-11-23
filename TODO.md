# TODO

The human prompt, verbatim: USER, live 2026-10-19, in
[#134](https://github.com/rel-int/discopy-next/issues/134):

> ha yeah that was a mistake from my side, what you wrote makes sense and what
> i had in mind was not EM but the free algebra-enrichment over a monoidal
> category where arrows really are M(A -> B) not sure where that would fit?
> discopy.enriched could be a cool module

Slice 3 of the plan in #134, verbatim:

> `discopy.enriched`: `Arrow[C, M]`, tensor (monoidal iff M commutative),
> `to_channel`, extension M\*C → D into algebra-enriched D, list case against
> `cat.Sum`

and the two defaults of its ask
([comment](https://github.com/rel-int/discopy-next/issues/134#issuecomment-7021172442)),
taken 2026-11-23 with no word from USER:

> 1. **How a functor extends to `M*C`.** A functor `F: C -> D` extends to
>    `M*C -> D` when each hom of `D` is an `M`-algebra and composition in `D`
>    is a homomorphism in each variable. Default: `D` gives that algebra as
>    one classmethod, `D.algebra(M, dom, cod)`, which turns `M(D(A, B))` into
>    `D(A, B)`. `cat.Arrow` implements it with `cat.Sum` for the list monad.
>    The other option is a per-functor argument `Functor(..., algebra=...)`.
> 2. **The list monad.** There is none in `kleisli.monad`, and `cat.Sum` as
>    the list case needs one. Default: add `List` to `kleisli.monad` in this
>    slice, beside the monads already there.

- [ ] `kleisli.monad.List`
- [ ] `D.algebra(M, dom, cod)` on `cat.Arrow` (the sum, `cat.Sum`) and `matrix.Matrix`
- [ ] `discopy/enriched.py`: `Arrow[category, monad]` with `id`, `then` and `tensor`, `to_channel`, `Functor` extending a functor along `D.algebra`
- [ ] `test/enriched.py`, with values derived by hand
- [ ] docs: `docs/api/semantics.rst`, `discopy/__init__.py`, `CHANGELOG.md`
- [ ] checks: pytest, pflake8, pylint, ruff, ty
