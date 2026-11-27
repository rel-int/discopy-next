`frobenius.Wire` reads its winding number modulo its one winding, i.e. as
zero, since a self-dual object has nothing to wind around: `Wire('x', 1)` is
`Wire('x')`, whether it is built or unpickled, `Wire('x', 1).l.l` is
`Wire('x')` and a functor applied to a hand-built wire with a nonzero `z`
returns instead of recursing until the stack runs out ([next#694](https://github.com/rel-int/discopy-next/pull/694)).
