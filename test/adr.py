"""Check docs/adr/: every record is dated, and every link to one resolves."""

import datetime
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ADR_DIR = ROOT / "docs" / "adr"
RECORD = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2})-[a-z0-9]+(?:-[a-z0-9]+)*\.md")
ACCEPTED = re.compile(r"^Accepted, (\d{4}-\d{2}-\d{2})", re.M)
LINK = re.compile(r"adr/(\d{4}[\w.-]*)")
LOCAL = re.compile(r"\]\(([^)/#]+\.md)\)|`([^`/]+\.md)`")
SUFFIXES = {".md", ".py", ".rst", ".toml", ".yml", ".yaml", ".lean", ".typ",
            ".txt", ".jinja"}
SKIP = {".git", ".amp", ".venv", "_build", "node_modules", ".lake",
        "__pycache__"}


def record_problems(names, read=lambda name: ""):
    """What is wrong with the file names of docs/adr/ and their dates."""
    problems = []
    for name in names:
        if name == "index.md":
            continue
        match = RECORD.fullmatch(name)
        if not match:
            problems.append(f"{name}: not YYYY-MM-DD-slug.md")
            continue
        try:
            datetime.date.fromisoformat(match.group(1))
        except ValueError:
            problems.append(f"{name}: {match.group(1)} is not a date")
            continue
        accepted = ACCEPTED.search(read(name))
        if accepted and accepted.group(1) != match.group(1):
            problems.append(
                f"{name}: accepted {accepted.group(1)}, not on its date")
    return problems


def references(path, text):
    """The record names a file links to: by path anywhere, by file name
    alone in docs/adr/ itself."""
    names = LINK.findall(text)
    if path.parent == ADR_DIR:
        names += [a or b for a, b in LOCAL.findall(text)]
    return names


def dangling(path, text, records):
    return [name for name in references(path, text)
            if name not in records and name + ".md" not in records]


def files(root):
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if d not in SKIP]
        for name in names:
            path = Path(directory) / name
            if path.suffix in SUFFIXES and path != Path(__file__).resolve():
                yield path


def test_records_are_dated():
    names = sorted(path.name for path in ADR_DIR.iterdir())
    assert "index.md" in names and len(names) >= 10
    read = (lambda name: (ADR_DIR / name).read_text(encoding="utf-8"))
    assert record_problems(names, read) == []


def test_links_to_records_resolve():
    records = {path.name for path in ADR_DIR.glob("*.md")}
    found = {}
    for path in files(ROOT.resolve()):
        text = path.read_text(encoding="utf-8", errors="replace")
        if bad := dangling(path, text, records):
            found[str(path.relative_to(ROOT.resolve()))] = bad
    assert found == {}


@pytest.mark.parametrize("name, problem", [
    ("0008-a-type-of-its-level.md", "not YYYY-MM-DD-slug.md"),
    ("2026-11-2-short-day.md", "not YYYY-MM-DD-slug.md"),
    ("2026-11-20-Upper.md", "not YYYY-MM-DD-slug.md"),
    ("2026-11-20_underscore.md", "not YYYY-MM-DD-slug.md"),
    ("2026-11-20-trailing-.md", "not YYYY-MM-DD-slug.md"),
    ("2026-11-20-record.txt", "not YYYY-MM-DD-slug.md"),
    ("2026-11-20-record.md\n", "not YYYY-MM-DD-slug.md"),
    ("\u0662026-11-20-record.md", "not YYYY-MM-DD-slug.md"),
    ("2026-02-30-no-such-day.md", "is not a date"),
    ("2026-11-21-late.md", "accepted 2026-11-20, not on its date"),
])
def test_malformed_record(name, problem):
    read = (lambda _: "## Status\n\nAccepted, 2026-11-20.\n")
    assert [problem in p for p in record_problems([name], read)] == [True]


def test_superseded_record_has_no_date_to_check():
    read = (lambda _: "## Status\n\nSuperseded by `2026-11-17-b.md`.\n")
    assert record_problems(["index.md", "2026-11-16-a.md"], read) == []


def test_dangling_links():
    records = {"index.md", "2026-11-16-a.md", "2026-11-17-b.md"}
    elsewhere, here = ROOT / "CONTRIBUTING.md", ADR_DIR / "2026-11-17-b.md"
    text = ("see `docs/adr/2026-11-16-a.md`, docs/adr/0004 and "
            "[old](docs/adr/0006-typst.md), and docs/adr/index.md")
    assert dangling(elsewhere, text, records) == ["0004", "0006-typst.md"]
    text = ("[`2026-11-16-a.md`](2026-11-16-a.md) made it, "
            "superseded by [0007](0007-latex.md), `0008-gone.md`")
    assert dangling(here, text, records) == ["0007-latex.md", "0008-gone.md"]
    assert dangling(elsewhere, "[x](0007-latex.md)", records) == []
