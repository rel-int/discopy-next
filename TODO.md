# TODO

[next#463](https://github.com/rel-int/discopy-next/issues/463), quoted verbatim:

> `python.multiplicative.Function.tensor`, `swap` and `copy` build `Function(...)` directly, while `id` and `permutation` build `cls(...)`. So a subclass of `python.Function` can't be a functor's codomain: `id` gives the subclass, `copy` gives a plain `Function`, and `then` refuses to compose the two.
>
> ```python
> from discopy import markov, python
> G = markov.Ty('G')
> class F(python.Function): pass
> func = markov.Functor({G: tuple}, {}, cod=F)
> func(markov.Copy(G))
> # TypeError: Expected __main__.F, got python.multiplicative.Function instead.
> ```
>
> Found while evaluating `markov.Bubble`s through a functor, for rel-int/wiki#87 (Boltzmann samplers over dsl1d diagrams). That wiki page works around it by handling bubbles in a `markov.Functor` subclass, so nothing waits on this.
>
> Fix: `swap`, `copy` (and `discard` if it does the same) become classmethods returning `cls(...)`, and `tensor` returns `type(self)(...)`, the way `permutation` already does.

## Work

- [ ] `multiplicative.Function`: `swap`, `copy`, `discard` and `ev` become classmethods
      returning `cls(...)`, `tensor`, `curry` and `fix` return `type(self)(...)`, `uncurry`
      calls `self.ev`. `discopy.abc` already declares every one of them a classmethod.
- [ ] `additive.Function`: the same defect at `tensor`, `swap`, `trace`, `dagger` and
      `merge`, plus `twist = id` inside the class body instead of a bound `Function.id`.
- [ ] `finset.Function`: decide and record whether it has the defect — `Permutation`
      subclasses it narrowing `__init__`, so `cls(...)` is not a safe refactor there.
- [ ] Check every call site of the changed methods in `discopy/`, `test/` and `proptest/`.
- [ ] The issue's reproducer as a test in `test/python/multiplicative.py`, plus the
      `additive` equivalent in `test/python/additive.py`.
- [ ] `CHANGELOG.md` bullet under `[Unreleased]` with a `[next#463]` reference link.
- [ ] Gates: `pflake8`, `pylint`, `ruff`, `ty`, `coverage run -m pytest`, each as a delta
      against `origin/main`.
- [ ] Report the `uncurry` signature mismatch with `abc` (`abc` takes `n`, `multiplicative`
      does not) as its own issue.
