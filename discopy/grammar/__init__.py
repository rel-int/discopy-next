# -*- coding: utf-8 -*-

"""
DisCoPy's grammar modules.

:mod:`discopy.grammar.neural` needs PyTorch, which the ``grammar`` extra
installs, so it is not imported here: import it on its own.
"""

from discopy.grammar import (
    thue, cfg, categorial, pregroup, dependency, abstract, proofnet)
from discopy.grammar.thue import Word, Rule
