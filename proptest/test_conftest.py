"""
Unit tests for the profiles of :mod:`proptest.conftest`: which repository's
example database the ``shared`` profile replays.
"""

from hypothesis.database import GitHubArtifactDatabase

from proptest.conftest import artifact


def test_shared_replays_upstream_by_default(monkeypatch):
    """ Unset, the variable leaves the file as it is upstream. """
    monkeypatch.delenv("HYPOTHESIS_GITHUB_REPOSITORY", raising=False)
    assert artifact() == GitHubArtifactDatabase("discopy", "discopy")


def test_shared_replays_the_repository_the_environment_names(monkeypatch):
    """ As this repo's CI and CONTRIBUTING.md set it. """
    monkeypatch.setenv("HYPOTHESIS_GITHUB_REPOSITORY", "rel-int/discopy-next")
    assert artifact() == GitHubArtifactDatabase("rel-int", "discopy-next")
