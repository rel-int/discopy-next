syntax
======

DisCoPy's mathematical core: a zoo of diagrams, categories and functors,
and the property tests that check their axioms.

.. raw:: html
    :file: architecture.html

The diagram shows the syntax, the free categories of each level. Going down a
column adds symmetry, going right along a row adds structure: ``feedback`` a
delay and a trace, ``closed`` exponentials, ``feedback.closed`` both, and the
last column duals, traced with the identity as delay; ``linear`` beside the
plain levels adds a bang. A module imports the ones above it in its column and
to its left on its row, except that ``linear`` and ``closed`` import nothing
from ``feedback`` and ``feedback`` nothing from ``linear``; forgetful functors
between the categories go the other way. Each module is pictured by its
characteristic generator, drawn by the module itself: ``docs/architecture.py``
draws the pictures and writes the table.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.testable
    discopy.abc
    discopy.lean
    discopy.cat
    discopy.monoidal
    discopy.spacial
    discopy.braided
    discopy.feedback
    discopy.balanced
    discopy.symmetric
    discopy.markov
    discopy.comarkov
    discopy.closed
    discopy.linear
    discopy.rigid
    discopy.pivotal
    discopy.ribbon
    discopy.compact
    discopy.frobenius
    discopy.hypergraph
    discopy.cmap

Utilities shared by every module: serialisation, factories and configuration.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.utils
    discopy.config
