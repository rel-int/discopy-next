# -*- coding: utf-8 -*-

"""
DisCoPy's grammar modules.

:mod:`~discopy.lambek.abstract.neural` and
:mod:`~discopy.lambek.categorial.bobcat` need PyTorch, which the ``lambek``
extra installs, so they are not imported here: import them on their own.
"""

from discopy.lambek import (
    thue, cfg, categorial, pregroup, dependency, abstract)
from discopy.lambek.thue import Word, Rule
