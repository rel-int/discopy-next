`quantum.zx.Hypergraph.size` counts two spiders as joined when an odd number of
Hadamards lie between them, as `to_pyzx` does: a Hadamard self-loop is a
phase and two parallel Hadamards cancel ([next#349](https://github.com/rel-int/discopy-next/issues/349)).
