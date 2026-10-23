"""
Optyx: ZX-based software for networked quantum computing, i.e. string
diagrams mixing qubits, photonic modes and classical bits, with lossy
channels, heralded measurements and classical feed-forward, evaluated as
tensor networks or with permanents.

:mod:`discopy.optyx.channel` is the category of channels these diagrams live
in, with qubits in :mod:`discopy.optyx.qubits`, photons in
:mod:`discopy.optyx.photonic` and bits in :mod:`discopy.optyx.classical`, on
top of the calculi of :mod:`discopy.optyx.core`, and
:mod:`discopy.optyx.compiler` compiles measurement-based patterns to
single-emitter photonic hardware.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.optyx.channel
    discopy.optyx.qubits
    discopy.optyx.photonic
    discopy.optyx.classical
    discopy.optyx.interaction
    discopy.optyx.core
    discopy.optyx.compiler

Note
----
Unlike the rest of DisCoPy, which is under the BSD 3-Clause License,
``discopy.optyx`` is under the Apache License, Version 2.0: see ``LICENSE``
and ``NOTICE`` in its directory. It needs the ``optyx`` extra, i.e.
``pip install discopy[optyx]``, and ``import discopy.optyx`` imports none of
perceval, graphix, pytket or qiskit until they are called for.
"""

from discopy.optyx import channel
from discopy.optyx.core import zw
from discopy.optyx.core import zx
from discopy.optyx.core import path
from discopy.optyx.core import backends
from discopy.optyx.core import diagram
from discopy.optyx.core import control
from discopy.optyx.channel import (
    Channel,
    CQMap,
    Discard,
    Encode,
    Measure,
    Diagram,
    Swap,
    Spider,
    Id,
    Scalar,
    mode,
    qmode,
    qubit,
    bit
)
