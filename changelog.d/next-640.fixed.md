The empty type of a colour is its own adjoint and reads back from its
`repr` below `monoidal`: `rigid.Ty.l` and `.r` kept the colour of a type
with no object to read it off, and `biclosed.Ty.__repr__` was
`monoidal.Ty.__repr__` without the branch that prints `Ty.id(colour)`, so
`Ty.id(Colour('black'))` printed as `rigid.Ty()` from `biclosed` down to
`frobenius` ([next#640](https://github.com/rel-int/discopy-next/pull/640)).
