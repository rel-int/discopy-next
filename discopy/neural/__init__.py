# -*- coding: utf-8 -*-

"""
``discopy.neural`` interprets DisCoPy diagrams as neural networks:
:mod:`discopy.neural.network` is the traced category of feedforward
networks, with lists of shapes :class:`Dims` as objects and layers as boxes,
compiling to a function on tensors that :mod:`discopy.neural.torch` runs as
a PyTorch module and ``jax.jit`` takes as it is.
:mod:`discopy.neural.signature` says what a generator of a source category
promises: its legs grouped into orbits, the symmetry each orbit carries, and
the wirings drawn out of a family's combinatorics.

The rest is recursive message passing, the networks of *Tiny Recursive
Message Passing* (arXiv:2610.18649): :mod:`discopy.neural.wiring` reads the
tables a network gathers along off a factor graph, and the blocks of a group
action off a wiring invariant under it; :mod:`discopy.neural.cell` holds the
boxes, a site on every node and a Deep Sets unit on every factor;
:mod:`discopy.neural.recursion` fires them in rounds, cycles and supervision
steps with a detached prefix; :mod:`discopy.neural.lattice` and
:mod:`discopy.neural.act` read them out by lattice deduction or by adaptive
computation time, each with its trainer; :mod:`discopy.neural.train` holds
the optimiser, the schedule, the weight average and the per-sample clip;
:mod:`discopy.neural.geometric` runs a network along the blocks of its
symmetry group rather than along its tables.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.network
    discopy.neural.signature
    discopy.neural.torch
    discopy.neural.wiring
    discopy.neural.cell
    discopy.neural.recursion
    discopy.neural.lattice
    discopy.neural.act
    discopy.neural.train
    discopy.neural.geometric

Note
----
``import discopy.neural`` imports no tensor framework: networks are built,
composed and rewired without one, and so are wirings. The recursive
networks are PyTorch modules: import their modules by name.
"""

from discopy.neural import network, signature
from discopy.neural.network import Dim, Dims, Network
from discopy.neural.signature import (
    Orbit, Signature, Sym, from_incidence, from_relation)

__all__ = [
    "Dim", "Dims", "Network", "Orbit", "Signature", "Sym", "from_incidence",
    "from_relation", "network", "signature"]
