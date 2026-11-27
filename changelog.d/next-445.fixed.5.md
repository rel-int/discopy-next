`neural.geometric.Shift` with a step longer than the grid's side reads zeros,
the translation its docstring defines, instead of raising in `narrow()`
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
