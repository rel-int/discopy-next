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
    symmetric
    markov

:mod:`linear.cat <discopy.linear.cat>` is the comonad, the
:class:`cat.Dereliction` ``!x -> x`` and the :class:`cat.Digging`
``!x -> !!x``, which need no tensor.
:mod:`linear.monoidal <discopy.linear.monoidal>` adds what does: the
:class:`monoidal.Contraction` and :class:`monoidal.Weakening` of the comonoid
on a banged object, the :class:`monoidal.Laxator` and the
:class:`monoidal.Exchange`, which makes a banged object central rather than
the category symmetric. :mod:`~discopy.linear.symmetric` takes the swap
for the exchange and :mod:`~discopy.linear.markov` the copy and discard of a
banged object for its contraction and weakening. Linear logic and
:mod:`~discopy.feedback` do not import each other: they meet in
:class:`~discopy.abc.GoISituation`.
"""

from discopy.linear import cat, monoidal, symmetric, markov  # noqa: F401
