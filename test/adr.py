"""Check that docs/adr/ record filenames are well-formed and distinct."""

import collections
import re
from pathlib import Path

import pytest

ADR_DIR = Path(__file__).parent.parent / "docs" / "adr"
RECORD = re.compile(r"^([0-9]{4})-[a-z0-9-]+\.md$")


def adr_numbers(paths):
    names = [Path(path).name for path in paths
             if Path(path).name != "index.md"]
    bad = [name for name in names if not RECORD.match(name)]
    if bad:
        raise ValueError(f"Malformed ADR filename(s): {bad}")
    by_number = collections.defaultdict(list)
    for name in names:
        by_number[RECORD.match(name).group(1)].append(name)
    duplicates = {num: files for num, files in by_number.items()
                  if len(files) > 1}
    if duplicates:
        raise ValueError(f"Duplicate ADR number(s): {duplicates}")
    return list(by_number)


def test_adr_numbers():
    paths = sorted(ADR_DIR.glob("*.md"))
    assert ADR_DIR / "index.md" in paths
    numbers = adr_numbers(paths)
    assert numbers == sorted(numbers)
    assert len(numbers) >= 9


def test_adr_duplicate_number(tmp_path):
    for name in (
        "index.md",
        "0001-first.md",
        "0008-a-type-of-its-level.md",
        "0008-contributions-belong-to-the-association.md",
    ):
        (tmp_path / name).write_text("# Record\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Duplicate ADR number.*0008"):
        adr_numbers(sorted(tmp_path.glob("*.md")))


def test_adr_malformed_filename(tmp_path):
    for bad in (
        "8-short.md",
        "0008_underscore.md",
        "0008-Upper.md",
        "adr.md",
        "\u0660\u0660\u0660\u0661-record.md",
    ):
        with pytest.raises(ValueError, match="Malformed ADR filename"):
            adr_numbers([tmp_path / "index.md", tmp_path / bad])
