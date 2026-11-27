Rotation and controlled-rotation gates compare and hash their phase exactly,
and the repr of every parametrized gate prints its phase exactly, a complex,
a fraction or a large integer included, so `eval(repr(g)) == g`, while `str`
and drawings keep 3 significant digits: a dictionary or functor keyed on
`Rz(0.1231)` no longer maps `Rz(0.1234)` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
