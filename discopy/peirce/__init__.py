"""
Peirce's existential graphs as two-coloured string diagrams, see
:cite:t:`BonchiEtAl24`: the white region is :class:`discopy.tensor.Tensor`
of ``bool`` as it is, the black region its elementwise negation, and a cut is
where the colour changes.

Summary
-------

.. autosummary::
    :template: module.rst
    :nosignatures:
    :toctree:

    alpha
    beta
"""

from discopy.peirce import alpha, beta

__all__ = ["alpha", "beta"]
