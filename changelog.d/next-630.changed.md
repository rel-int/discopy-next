The dagger of an `optyx.core` box defined by its array keeps its name and
flips `is_dagger`, as `cat.Box.dagger` does, so `f[::-1][::-1] == f`; a
box with `is_dagger=None` is self-adjoint and is its own dagger. A dagger
prints `f[::-1]`, not `f.dagger()`. Truncating a dagger gives the dagger
of the truncated box, and its repr prints the array, so it reads back. An
array with an axis per wire is daggered wire by wire, where it used to be
transposed as a matrix, which reversed the order of its wires
([next#630](https://github.com/rel-int/discopy-next/pull/630)).
