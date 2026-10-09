"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.
"""

import pytest
from _pytest.doctest import DoctestItem


UNIMPORTABLE = ("discopy/quantum/pennylane.py", "discopy/quantum/tk.py")


def pytest_configure(config):
    """
    Register the doctests of the rules: a method decorated
    :func:`discopy.search.rule` is a :class:`discopy.pattern.Declaration`
    in its class's dict, which doctest's finder does not recurse into,
    so each module lists the functions they wrap in its ``__test__``.
    """
    import importlib
    import pkgutil
    import sys

    import discopy
    from discopy.pattern import Declaration

    for info in pkgutil.walk_packages(discopy.__path__, "discopy."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            continue
    for name, module in list(sys.modules.items()):
        if not name.startswith("discopy"):
            continue
        tests = {}
        for cls in list(vars(module).values()):
            if not isinstance(cls, type) or cls.__module__ != name:
                continue
            for attr, value in vars(cls).items():
                while isinstance(value, (staticmethod, classmethod)):
                    value = value.__func__
                if (isinstance(value, Declaration)
                        and value.function.__doc__
                        and value.function.__module__ == name
                        and value.function.__name__ == attr):
                    tests[f"{cls.__qualname__}.{attr}"] = value.function
        if tests:
            module.__test__ = dict(
                getattr(module, "__test__", {}), **tests)


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--skip-extra"):
        return None
    return collection_path.as_posix().endswith(UNIMPORTABLE) or None


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped. """
    if not config.getoption("--skip-extra"):
        return
    for item in items:
        if isinstance(item, DoctestItem) and item.dtest is not None and any(
                "+EXTRA" in e.source for e in item.dtest.examples):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))
