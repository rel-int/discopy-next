`monoidal.Diagram.encode` gives its domain and one list of `(box, offset)`
per layer, so that `decode(*d.encode()) == d` on a foliated diagram too;
`decode` reads that or a flat list, one box per layer, and
`to_staircases` flattens the layers itself. This breaks a caller that
reads the second element of `encode` as flat `(box, offset)` pairs: the
old list is `list(zip(d.boxes, d.offsets))`, which `decode` still reads,
one box per layer ([next#629](https://github.com/rel-int/discopy-next/pull/629)).
