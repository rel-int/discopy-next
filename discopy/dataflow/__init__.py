# -*- coding: utf-8 -*-

"""
Dataflow diagrams, i.e. feedback diagrams whose wires carry streams with a
head and a tail, after :cite:t:`DiLavoreEtAl22`.

A wire ``x`` has a head, its first value, and a tail, the values after it,
which is ``x`` itself when ``x`` is constant. A diagram has a head and a
tail, and its followed-by is the isomorphism from ``x.head @ x.tail.delay()``
to ``x``, the ``fby`` of Lucid.

The structure lives in the base level,
:mod:`dataflow.symmetric <discopy.dataflow.symmetric>` over
:mod:`feedback.symmetric <discopy.feedback.symmetric>`, since monoidal
streams are defined over symmetric categories: a level above it, e.g.
:mod:`dataflow.markov <discopy.dataflow.markov>` over
:mod:`feedback.markov <discopy.feedback.markov>`, mixes each of its classes
into the class of the feedback level it extends. See :mod:`~discopy.stream`
for their interpretation as streams.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    symmetric
    markov
"""

from discopy.dataflow import symmetric, markov  # noqa: F401
