"""Check that the distribution declares the license of every file it bundles:
the root LICENSE for DisCoPy, and LICENSE and NOTICE in discopy/optyx."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

with open(ROOT / "pyproject.toml", "rb") as file:
    PROJECT = tomllib.load(file)["project"]


def test_license_is_an_spdx_expression():
    assert PROJECT["license"] == "BSD-3-Clause AND Apache-2.0"


def test_every_license_file_is_declared():
    bundled = {"LICENSE"} | {
        path.relative_to(ROOT).as_posix()
        for pattern in ("LICENSE*", "NOTICE*")
        for path in (ROOT / "discopy").rglob(pattern)}
    declared = {
        path.relative_to(ROOT).as_posix()
        for pattern in PROJECT["license-files"]
        for path in ROOT.glob(pattern)}
    assert declared == bundled
