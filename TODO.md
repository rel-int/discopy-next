fix https://github.com/rel-int/discopy-next/issues/635 (a next-repo bug, no design decision; the issue's own default is "an agent makes the fix in its own PR").

Issue text: `biclosed.Constant.__init__` sets `dom=self.ob()`, the empty type with no colour, whatever the colour of `cod`. A constant of a type of coloured wires then fails:
```python
from discopy.monoidal import Colour
from discopy import frobenius
white = Colour("white")
x = frobenius.Ty(frobenius.Wire("x", dom=white, cod=white))
x.constant_factory("f", x)
# AxiomError: Expected a globular box, got boundaries None -> None and white -> white.
```

- [x] a test that fails before the fix
- [x] `Constant` takes the empty type of the colour of its type as domain
- [x] CHANGELOG.md bullet
