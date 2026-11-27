`optyx.core.control.ControlledPhaseShift.conjugate` negates its phases: it
returned a box built from the arguments of `self`, so the conjugate *was*
`self` and its truncation equalled the original's. Complex conjugation sends
`exp(2*pi*1j*y)` to `exp(-2*pi*1j*y)`, so the conjugate is the same box with
every value of its function negated, and `conjugate` now commutes with
`dagger` and is an involution. The negation is `partial(negate_phases,
function)`, a function of the module rather than a closure, so a conjugated
box pickles whenever the box it is the conjugate of does ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
