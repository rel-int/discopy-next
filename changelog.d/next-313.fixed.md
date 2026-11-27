`quantum.zx.Diagram.from_pyzx`, and its copy in `optyx.core.zx`, wires a
spider with several outputs to the outputs it is joined to: it used to move
the leg already placed at an earlier output, so the tensor differed from
PyZX's ([next#313](https://github.com/rel-int/discopy-next/issues/313)).
