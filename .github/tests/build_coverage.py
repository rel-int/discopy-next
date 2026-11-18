"""Tests for the Coverage step of ``build``, run as GitHub runs it.

The controller of ``pytest -n auto`` writes a data file of its own and each
xdist worker one more: a combine that finds the controller's alone measured
no test, and the step says so instead of reading as a coverage regression
(#596). Each test writes the data files of processes that each ran a script
of its own, since ``combine`` skips a file whose data it already has, then
runs the step's shell with ``uv run`` standing for the Python under test.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def step():
    with open(GITHUB / "workflows" / "build.yml") as file:
        steps = yaml.safe_load(file)["jobs"]["test"]["steps"]
    (step, ) = (step for step in steps if step.get("name") == "Coverage")
    assert step["shell"] == "bash"
    return step["run"]


def run(step, tmp_path, processes):
    for i in range(processes):
        (tmp_path / f"script{i}.py").write_text("x = 1\n")
        subprocess.run([sys.executable, "-m", "coverage", "run",
                        "--parallel-mode", f"script{i}.py"],
                       cwd=tmp_path, check=True)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uv").write_text(
        f'#!/bin/sh\nshift\nshift\nexec "{sys.executable}" -m coverage "$@"\n')
    (fake / "uv").chmod(0o755)
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path),
               PATH=f"{fake}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step],
        cwd=tmp_path, env=env, capture_output=True, text=True)


def test_the_step_reports_the_controller_and_its_workers(step, tmp_path):
    result = run(step, tmp_path, processes=3)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Combining data file") == 3
    assert "script2.py" in result.stdout


def test_the_controller_alone_fails_as_its_own_error(step, tmp_path):
    result = run(step, tmp_path, processes=1)
    assert result.returncode == 1
    assert "::error::coverage combine merged 1 data file(s)" in result.stdout
    assert "script0.py" not in result.stdout
