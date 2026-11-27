`python.multiplicative.Function` and `python.additive.Function` build their
structural morphisms with `cls` and `type(self)` rather than with the base
class, the way `permutation` already did, so a subclass can be a functor's
codomain: `swap`, `copy`, `discard`, `ev`, `merge` and `twist` are the
classmethods `discopy.abc` declares them to be, and `tensor`, `curry`,
`fix`, `trace` and `dagger` return the class of their argument. A
`markov.Functor` into a subclass of `python.Function` used to send `Copy` to
a plain `Function` and then raise `TypeError: Expected F, got
python.multiplicative.Function instead` on composing it with the `Id` the
subclass did give. A subclass that is also a symmetric functor's codomain
wants `@factory` so that its `ar` is itself rather than its base, see the
second gotcha of the README. `python.finset.Function` keeps its base class
on purpose: `finset.Permutation` subclasses it narrowing `__init__`
([next#463](https://github.com/rel-int/discopy-next/issues/463)).
