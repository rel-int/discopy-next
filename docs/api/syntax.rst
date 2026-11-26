syntax
======

DisCoPy's mathematical core: a zoo of diagrams, categories and functors,
and the property tests that check their axioms.

.. raw:: html
    :file: architecture.html

The diagram shows the syntax, the free categories of each level: software
dependencies between their modules go top-to-bottom, left-to-right and
forgetful functors between the categories go the other way.

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
    balanced
    symmetric
    markov
    comarkov
    closed
    biclosed
    bang
    rigid
    pivotal
    ribbon
    compact
    frobenius
    hypergraph
    cmap

Utilities shared by every module: serialisation, factories and configuration.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    utils
    config
