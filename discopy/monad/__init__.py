# -*- coding: utf-8 -*-

"""
DisCoPy's monad modules: free and algebra.

The monads themselves are :class:`discopy.kleisli.monad.Monad`, and their
Kleisli categories are :mod:`discopy.kleisli`.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    free
    algebra
"""

from discopy.monad import free, algebra
from discopy.monad.algebra import Algebra
