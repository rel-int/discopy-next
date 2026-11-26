syntax
======

DisCoPy's mathematical core: a zoo of diagrams, categories and functors,
and the property tests that check their axioms.

.. raw:: html
    :file: architecture.html

The diagram shows the syntax, the free categories of each level: software
dependencies between their modules go top-to-bottom, left-to-right and
forgetful functors between the categories go the other way.

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
    discopy.biclosed
    discopy.bang
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
