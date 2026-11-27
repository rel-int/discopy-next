`matrix.asarray` builds an array from nested entries by stacking the ones
the active backend already owns, so `Matrix([t], 1, 1)` keeps the gradient
of a PyTorch tensor `t` where converting it entry by entry read it as a
scalar and detached it; `optyx.core.zw.Endo` keeps a scalar its backend
owns rather than calling `complex` on it, and conjugates it in that
backend, so a photonic gate's `to_path()` is differentiable like its
`array` — under the `matrix.backend` its parameters belong to, outside
which it now raises where it used to return a detached array
([next#654](https://github.com/rel-int/discopy-next/pull/654)).
