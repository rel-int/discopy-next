# -*- coding: utf-8 -*-

"""
The deprecated name of :mod:`discopy.lambek`.

Importing it warns. Every name of ``discopy.grammar`` is that of
:mod:`discopy.lambek`, and ``discopy.grammar.x`` is the module
``discopy.lambek.x``, imported when it is asked for, so that code and pickles
written under the old name still load.

>>> import sys, warnings
>>> from importlib import import_module, reload
>>> from discopy import lambek
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     grammar = reload(import_module("discopy.grammar"))
>>> print(caught[-1].message)
discopy.grammar is deprecated, use discopy.lambek instead.
>>> assert grammar.pregroup is lambek.pregroup
>>> from discopy.grammar import Word
>>> from discopy.grammar.cfg import Rule
>>> assert Word is lambek.Word and Rule is lambek.cfg.Rule

A tree written under the old name loads with no import of this module, see
:data:`discopy.utils.RENAMED`.

>>> from discopy.utils import dumps, loads
>>> n = lambek.pregroup.Ty('n')
>>> old = dumps(n).replace("lambek.pregroup", "grammar.pregroup")
>>> assert loads(old) == n
"""

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

from discopy import lambek

warnings.warn(
    "discopy.grammar is deprecated, use discopy.lambek instead.",
    DeprecationWarning, stacklevel=2)

__path__: list[str] = []


class Alias(MetaPathFinder, Loader):
    """
    Finds ``discopy.grammar.x`` as the module ``discopy.lambek.x``, importing
    it if need be.
    """
    def find_spec(self, fullname, path=None, target=None):
        """ The spec of a submodule of ``discopy.grammar``, else ``None``. """
        # pylint: disable=unused-argument  # the name alone finds the module
        if fullname.startswith(__name__ + "."):
            return spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        """
        The module of :mod:`discopy.lambek` with the same name, its own spec
        kept as the state of the loader.
        """
        module = import_module(
            lambek.__name__ + spec.name.removeprefix(__name__))
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        """
        Nothing to run, the module was imported under its own name: it gets
        its own spec back, so that it still reloads.
        """
        module.__spec__ = module.__spec__.loader_state


if not any(isinstance(finder, Alias) for finder in sys.meta_path):
    sys.meta_path.append(Alias())


def __getattr__(name):
    return getattr(lambek, name)
