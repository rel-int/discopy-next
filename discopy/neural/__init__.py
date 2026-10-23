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

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.network
    discopy.neural.signature
    discopy.neural.torch

Note
----
``import discopy.neural`` imports no tensor framework: networks are built,
composed and rewired without one.
"""

from discopy.neural import network, signature
from discopy.neural.network import Dim, Dims, Network
from discopy.neural.signature import (
    Orbit, Signature, Sym, from_incidence, from_relation)

__all__ = [
    "Dim", "Dims", "Network", "Orbit", "Signature", "Sym", "from_incidence",
    "from_relation", "network", "signature"]
