"""
Random symmetric and compact diagrams with a functor evaluating them as
tensors, shared by the tests of the decoders of :mod:`discopy.hypergraph` and
:mod:`discopy.cmap`.
"""

from math import prod
from random import Random

from discopy import compact, tensor
from discopy.tensor import Dim


def random_diagram(module, seed: int, n_layers: int = 8):
    """
    A random diagram of ``module``, i.e. :mod:`discopy.symmetric` or
    :mod:`discopy.compact`, built layer by layer from random boxes and
    permutations and, in the compact case, cups and caps, with a
    :class:`tensor.Functor` sending each box to a random array.
    """
    rng = Random(seed)
    atoms = {module.Ty(name): Dim(dim) for name, dim in zip("xyz", (2, 3, 2))}
    types, ar_map = list(atoms), {}

    def dim(typ):
        return Dim().tensor(*(atoms[module.Ty(a.inside[0].name)] for a in typ))

    def box(dom, cod):
        result = module.Box(f"f{len(ar_map)}", dom, cod)
        size = prod((dim(dom) @ dim(cod)).inside)
        ar_map[result] = [rng.randint(0, 3) for _ in range(size)]
        return result

    diagram = module.Diagram.id(module.Ty().tensor(*rng.choices(types, k=3)))
    for _ in range(n_layers):
        cod, choice = diagram.cod, rng.random()
        cups = [i for i in range(len(cod) - 1) if cod[i + 1] == cod[i].r]\
            if module is compact else []
        if choice < .3 and len(cod) > 1:
            perm = rng.sample(range(len(cod)), len(cod))
            diagram >>= module.Diagram.permutation(perm, cod)
        elif choice < .5 and cups:
            i = rng.choice(cups)
            diagram >>= cod[:i] @ module.Cup(cod[i], cod[i + 1]) @ cod[i + 2:]
        elif choice < .6 and module is compact:
            i, typ = rng.randint(0, len(cod)), rng.choice(types)
            diagram >>= cod[:i] @ module.Cap(typ.r, typ) @ cod[i:]
        else:
            i = rng.randint(0, len(cod))
            j = rng.randint(i, min(len(cod), i + 2))
            new = module.Ty().tensor(*rng.choices(types, k=rng.randint(0, 2)))
            diagram >>= cod[:i] @ box(cod[i:j], new) @ cod[j:]
    return diagram, tensor.Functor(atoms, ar_map, dom=module.Diagram)
