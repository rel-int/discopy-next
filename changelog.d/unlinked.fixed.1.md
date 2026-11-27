`optyx`: `d ** n` is the tensor power the hierarchy already states in
`monoidal.Ty.__pow__`, i.e. the identity for `n <= 0` on an endomorphism and
an assertion on anything else, where three copies of the same `__pow__` —
on `optyx.core.diagram.Box`, on `channel.Diagram` and a verbatim duplicate
of the latter on `channel.CQMap` — recursed without a floor for every
`n <= 0`, e.g. `zw.W(2) ** 0` and `photonic.BS ** 0`. A composite
`optyx.core.diagram.Diagram` has a power too, where only its boxes had one
