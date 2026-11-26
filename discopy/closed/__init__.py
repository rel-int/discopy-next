# -*- coding: utf-8 -*-

"""
Closed categories, i.e. monoidal categories with exponentials, a module for
each level of the hierarchy that has them.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    monoidal
    symmetric
    markov

:mod:`discopy.closed.monoidal` is the free biclosed category, with left and
right exponentials ``x << y`` and ``y >> x`` and terms that are planar and
linear. :mod:`discopy.closed.symmetric` is the free symmetric closed
category, where the two exponentials coincide. :mod:`discopy.closed.markov`
adds copy and discard, i.e. it is the free closed Markov category, with
lambda terms that need not be linear, tuples, projections, ``let`` and Böhm
trees. A :mod:`discopy.compact` diagram is a symmetric closed one, with
``x ⊸ y = x* @ y``, and a :mod:`discopy.rigid` diagram a biclosed one.
"""

from discopy.closed import monoidal, symmetric, markov  # noqa: F401
