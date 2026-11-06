"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.

A module of ``discopy`` is imported by its name before it is collected:
under ``--import-mode=importlib`` a module collected for its doctests while
its package is not yet imported is executed twice, once by the package's own
import and once by pytest, so a doctest of ``discopy/feedback.py`` run alone
reads a ``Ty`` that ``markov`` does not know, and one of
``discopy/optyx/core/path.py`` an ``Amplitudes`` that ``eval`` does not
return.
"""

import importlib
import pathlib

import pytest
from _pytest.doctest import DoctestItem

ROOT = pathlib.Path(__file__).resolve().parents[1]


UNIMPORTABLE = (
    "discopy/quantum/pennylane.py", "discopy/quantum/tk.py",
    "discopy/grammar/neural.py", "discopy/neural/torch.py",
    "discopy/neural/geometric.py",
    "discopy/neural/cell.py", "discopy/neural/recursion.py",
    "discopy/neural/lattice.py", "discopy/neural/act.py", "discopy/neural/train.py",
    "discopy/drawing/widget.py")

# A package whose own import needs an extra, e.g. ``discopy.optyx``.
UNIMPORTABLE_PACKAGES = ("discopy/optyx/", "test/optyx/")


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--skip-extra"):
        return None
    path = collection_path.as_posix()
    return path.endswith(UNIMPORTABLE) or any(
        package in path + "/" for package in UNIMPORTABLE_PACKAGES) or None


class Unimportable(pytest.File):
    """ A file of a package that needs an extra, given as an argument, so
    that ``pytest_ignore_collect`` is not asked about it. """
    def collect(self):
        pytest.skip("needs an extra")


def module_name(file_path):
    """ The dotted name of a module of ``discopy``, else ``None``. """
    path = pathlib.Path(file_path).resolve()
    if path.suffix != ".py" or not path.is_relative_to(ROOT / "discopy"):
        return None
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@pytest.hookimpl(wrapper=True)
def pytest_collect_file(file_path, parent):
    name = module_name(file_path)
    if name is not None:
        try:
            importlib.import_module(name)
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # pytest's own import of the file reports it
    collectors = yield
    if collectors and parent.config.getoption("--skip-extra") and any(
            package in file_path.as_posix()
            for package in UNIMPORTABLE_PACKAGES):
        return [Unimportable.from_parent(parent, path=file_path)]
    return collectors


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped. """
    if not config.getoption("--skip-extra"):
        return
    for item in items:
        if isinstance(item, DoctestItem) and item.dtest is not None and any(
                "+EXTRA" in e.source for e in item.dtest.examples):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))
