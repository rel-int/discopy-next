"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.
"""

import pytest
from _pytest.doctest import DoctestItem


UNIMPORTABLE = (
    "discopy/quantum/pennylane.py", "discopy/quantum/tk.py",
    "discopy/grammar/neural.py", "discopy/neural/torch.py",
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


@pytest.hookimpl(wrapper=True)
def pytest_collect_file(file_path, parent):
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
