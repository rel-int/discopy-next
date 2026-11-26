# -*- coding: utf-8 -*-

"""
The groups and actions shared by the tests of equivariant layers, and the
orbits of pairs of points by flood fill, independently of
:meth:`discopy.group.Action.orbits`.
"""

import itertools

import numpy

from discopy.group import Cyclic, Product, Symmetric, Wreath

TORUS = Product[Cyclic[3], Cyclic[3]]
LINES = Wreath[Symmetric[2], Symmetric[2]]
ACTIONS = (Cyclic[6].action(), Symmetric[4].action(), TORUS.action(),
           LINES.action(), Wreath[Cyclic[2], Symmetric[3]].action())


def brute(dom, cod):
    """
    The orbit of each pair ``(y, x)`` by flood fill along the generators,
    numbered in the order the pairs are met.
    """
    label = {}
    for start in itertools.product(range(cod.size), range(dom.size)):
        if start in label:
            continue
        label[start], stack = len(set(label.values())), [start]
        while stack:
            y, x = stack.pop()
            for g, h in zip(cod.images, dom.images):
                if (g[y], h[x]) not in label:
                    label[g[y], h[x]] = label[start]
                    stack.append((g[y], h[x]))
    return label


def brute_tying(dom, cod):
    """ The orbit numbers of :func:`brute` as a matrix indexed ``[y, x]``. """
    label = brute(dom, cod)
    return numpy.array([[label[y, x] for x in range(dom.size)]
                        for y in range(cod.size)])
