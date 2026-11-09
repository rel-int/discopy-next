"""Tests for merging the example databases of the shards of a proptest run."""

import pathlib


def shard(root, examples, deleted=()):
    for path, content in examples.items():
        (root / "examples" / path).parent.mkdir(parents=True, exist_ok=True)
        (root / "examples" / path).write_text(content)
    (root / "examples").mkdir(parents=True, exist_ok=True)
    (root / "deleted.txt").write_text("".join(f"./{p}\n" for p in deleted))
    return root


def files(root):
    return {path.relative_to(root).as_posix(): path.read_text()
            for path in root.rglob("*") if path.is_file()}


def test_union_of_what_the_shards_end_with(merge_examples, tmp_path):
    one = shard(tmp_path / "1", {"k/old": "o", "k/a": "a"})
    two = shard(tmp_path / "2", {"k/old": "o", "j/b": "b"})
    merge_examples.merge(tmp_path / "out", [one, two])
    assert files(tmp_path / "out") == {"k/old": "o", "k/a": "a", "j/b": "b"}


def test_a_deletion_is_not_brought_back_by_another_shard(
        merge_examples, tmp_path):
    one = shard(tmp_path / "1", {"j/kept": "x"}, deleted=["k/pruned"])
    two = shard(tmp_path / "2", {"j/kept": "x", "k/pruned": "p"})
    merge_examples.merge(tmp_path / "out", [one, two])
    assert files(tmp_path / "out") == {"j/kept": "x"}


def test_hidden_entries_are_kept(merge_examples, tmp_path):
    one = shard(tmp_path / "1", {".keys/k": "key"})
    merge_examples.merge(tmp_path / "out", [one])
    assert files(tmp_path / "out") == {".keys/k": "key"}


def test_a_shard_without_a_database(merge_examples, tmp_path):
    (tmp_path / "1").mkdir()
    merge_examples.merge(tmp_path / "out", [tmp_path / "1"])
    assert not pathlib.Path(tmp_path / "out").exists()
