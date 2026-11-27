Every box of `discopy.optyx` prints the call that built it, so that
`eval(repr(x)) == x` holds for it as STYLE.md asks and `testable.transparency`
states: the boxes of `channel`, `qubits`, `photonic` and `classical` narrow
`__init__` and used to print `cat.Box`'s `(name, dom, cod, data=...)`, which
their own constructor cannot take. `core.diagram.Mode` and `Bit` print their
count. A box holding a Python function — `channel.Observable` of a
`function`, `classical.ClassicalFunction`, `classical.BitControlledPhaseShift`
— prints that function as Python does, so it reads back only if the
function's representation does, as for `python.Function` ([next#351](https://github.com/rel-int/discopy-next/issues/351)).
