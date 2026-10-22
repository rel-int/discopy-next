# -*- coding: utf-8 -*-

"""
The free traced category, folded into :mod:`discopy.feedback`: a traced
category is a feedback category whose delay is the identity, and a
:class:`discopy.feedback.Trace` is a feedback whose memory is not delayed.

Every name of this module is a deprecated alias of the one in
:mod:`discopy.feedback`, kept for one release.
"""

from discopy.utils import deprecated_alias

# pylint: disable=undefined-all-variable
__all__ = (  # noqa: F822
    "Ty", "Diagram", "Box", "Trace", "Sum", "Functor", "CMap", "Hypergraph",
    "Id")

__getattr__ = deprecated_alias(__name__, {
    name: f"discopy.feedback.{name}" for name in __all__})
