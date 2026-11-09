"""
Unit tests for :mod:`proptest.conftest`: which repository's example
database the ``shared`` profile replays, and how ``--proptest-shard`` splits
the collection.
"""

from hypothesis.database import GitHubArtifactDatabase

from proptest.conftest import artifact, shard


def test_shared_replays_upstream_by_default(monkeypatch):
    """ Unset, the variable leaves the file as it is upstream. """
    monkeypatch.delenv("HYPOTHESIS_GITHUB_REPOSITORY", raising=False)
    assert artifact() == GitHubArtifactDatabase("discopy", "discopy")


def test_shared_replays_the_repository_the_environment_names(monkeypatch):
    """ As this repo's CI and CONTRIBUTING.md set it. """
    monkeypatch.setenv("HYPOTHESIS_GITHUB_REPOSITORY", "rel-int/discopy-next")
    assert artifact() == GitHubArtifactDatabase("rel-int", "discopy-next")


def test_the_shards_partition_the_collection():
    """ Every item in exactly one shard, in collection order within each. """
    items = [f"cell{i}" for i in range(100)]
    shards = [shard(items, index, 7) for index in range(1, 8)]
    assert sorted(sum(shards, [])) == sorted(items)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
