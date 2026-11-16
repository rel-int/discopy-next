# -*- coding: utf-8 -*-

"""
The deprecated name of :mod:`discopy.lambek`.

Importing it warns, and makes each ``discopy.grammar.x`` the module
``discopy.lambek.x`` that :mod:`discopy.lambek` imports, so that a pickle or
a tree written under the old name reads back.

>>> import warnings
>>> from importlib import reload
>>> from discopy import lambek
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     from discopy import grammar
...     grammar = reload(grammar)
>>> print(caught[-1].message)
discopy.grammar is deprecated, use discopy.lambek instead.
>>> assert grammar.pregroup is lambek.pregroup

>>> from discopy.utils import dumps, loads
>>> n = lambek.pregroup.Ty('n')
>>> old = dumps(n).replace("lambek.pregroup", "grammar.pregroup")
>>> assert loads(old) == n
"""

import sys
import warnings

from discopy import lambek

warnings.warn(
    "discopy.grammar is deprecated, use discopy.lambek instead.",
    DeprecationWarning, stacklevel=2)

for name in list(sys.modules):
    if name.startswith(lambek.__name__ + "."):
        child = name.removeprefix(lambek.__name__ + ".")
        sys.modules[f"{__name__}.{child}"] = sys.modules[name]
        if "." not in child:
            globals()[child] = sys.modules[name]
