# -*- coding: utf-8 -*-

"""
DisCoPy's grammar modules.

:mod:`discopy.lambek.neural` needs PyTorch, which the ``lambek`` extra
installs, so it is not imported here: import it on its own.
"""

from discopy.lambek import (
    thue, cfg, categorial, pregroup, dependency, abstract, proofnet)
from discopy.lambek.thue import Word, Rule
