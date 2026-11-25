# -*- coding: utf-8 -*-

"""
Linear logic, i.e. categories with a bang, the exponential ``!`` of
intuitionistic linear logic :cite:t:`Bierman95`, beside the hierarchy: a
module for each level the bang makes sense at.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    cat
    monoidal

:mod:`discopy.linear.cat` is the comonad, the :class:`cat.Dereliction`
``!x -> x`` and the :class:`cat.Digging` ``!x -> !!x``, which need no
tensor. :mod:`discopy.linear.monoidal` adds what does: the
:class:`monoidal.Contraction` and :class:`monoidal.Weakening` of the
comonoid on a banged object, the :class:`monoidal.Laxator` and the
:class:`monoidal.Exchange`, which makes a banged object central rather than
the category symmetric. Linear logic and :mod:`discopy.feedback` do not
import each other: they meet in :class:`discopy.abc.GoISituation`.
"""

from discopy.linear import cat, monoidal  # noqa: F401
