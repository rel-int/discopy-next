> so now Sum is just a subclass of Box with a special composition and drawing methods and a pointer sum_factory? i’d keep that in each module it
> doesn’t have much to fill a special module

> Go ahead, delete discopy.sum outright

> what is this formal_sum? sounds ugly let’s get rid of it

- [WIP] @T-14209019-2026-11-25 20:22 cat: `cat.Sum(Box)` holds the terms, `+`, `then`, `dagger`, `subs`/`lambdify`, the tree and the drawing
- [WIP] @T-14209019-2026-11-25 20:22 monoidal: `monoidal.Sum(cat.Sum, Box)` adds `tensor`; every module's `Sum` subclasses its parent module's, none `formal_sum`
- [WIP] @T-14209019-2026-11-25 20:22 delete `discopy/sum.py`, `Sum[...]`, the `formal_sum` alias and the `subscript_bases` hook if nothing else uses it
- [ ] tests, docs, notebooks and CHANGELOG follow; close #10 as superseded
- [ ] run next-checks
