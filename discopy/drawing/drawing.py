# -*- coding: utf-8 -*-

"""
The category of labeled progressive plane graphs, renamed
:mod:`~discopy.drawing.planegraph`.

Every name of this module is a deprecated alias of the one in
:mod:`~discopy.drawing.planegraph`, kept for one release.
"""

from discopy.utils import deprecated_alias

# pylint: disable=undefined-all-variable
__all__ = (  # noqa: F822
    "BOX_DRAWING_ATTRIBUTES",
    "TRANSPARENT",
    "Drawing",
    "Node",
    "PlaneGraph",
    "Point",
    "cross",
    "curve",
    "meet",
    "overlap",
    "sign",
    "through",
    "touch",
    "trailing_margin",
)

__getattr__ = deprecated_alias(__name__, {
    name: f"discopy.drawing.planegraph.{name}" for name in __all__})
