syntax
======

DisCoPy's mathematical core: a zoo of diagrams, categories and functors,
and the property tests that check their axioms.

.. raw:: html
    :file: architecture.html

The diagram shows the syntax, the free categories of each level. Going down a
column adds structure to the tensor: scalars that slide, braids, twists,
symmetry, then copies; going right along a row adds structure to the
category: ``feedback`` a delay and a trace, ``closed`` exponentials,
``feedback.closed`` both, and the last column duals, whose trace at delay
zero is a cap and a cup; ``linear`` beside the plain levels adds a bang, and
``dataflow`` beside ``feedback`` adds the head, tail and followed-by of streams. A module imports the
ones above it in its column, and ``feedback.closed`` imports ``feedback`` and
``closed`` on its row, but ``linear``, ``feedback`` and ``closed`` import
nothing from one another; forgetful functors between the categories go the
other way. Each module is pictured by its characteristic generator or equation,
drawn by the module itself: ``docs/architecture.py`` draws the pictures and writes the
table.

.. currentmodule:: discopy

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    testable
    abc
    lean
    cat
    monoidal
    spacial
    braided
    feedback
    dataflow
    balanced
    symmetric
    markov
    comarkov
    closed
    linear
    rigid
    pivotal
    ribbon
    compact
    frobenius
    hypergraph
    cmap
    inet

Utilities shared by every module: serialisation, factories and configuration.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    utils
    config
