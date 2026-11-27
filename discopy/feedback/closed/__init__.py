# -*- coding: utf-8 -*-

"""
Closed categories with feedback, a module for each level of
:mod:`~discopy.closed` and for :mod:`~discopy.feedback.balanced` between them,
whose wires carry a time step. Each is the free traced category over its
closed level once the delay is the identity, and the traced levels with
duals sit under them: :mod:`~discopy.pivotal` under ``monoidal``,
:mod:`~discopy.ribbon` under ``balanced``, :mod:`~discopy.compact` under
``symmetric`` and :mod:`~discopy.frobenius` under ``markov``.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    monoidal
    balanced
    symmetric
    markov
"""

from discopy.feedback.closed import (  # noqa: F401
    monoidal, balanced, symmetric, markov)
