"""Tests for what cancels a run of a workflow, or a job of one.

A concurrency group keeps one running and one pending run, and a newly
queued run cancels the pending one whatever ``cancel-in-progress`` says: a
workflow that lets every commit on main keep its run needs a group per
commit. One level down, ``fail-fast`` cancels the siblings of a job that
failed, which on main throws away the versions that were measuring the
commit.
"""

import ast
import json
import pathlib
import re

import pytest
import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent


class Context(dict):
    def __getattr__(self, name):
        value = self.get(name, {})
        return Context(value) if isinstance(value, dict) else value


def evaluate(expression, github):
    def value(match):
        code = match[1].replace("&&", " and ").replace("||", " or ")
        return str(eval(code, {
            "github": Context(github), "fromJSON": json.loads}))
    return re.sub(r"\$\{\{(.*?)\}\}", value, str(expression))


def versions(job, github):
    """The Pythons a job's matrix expands to, as the runner reads it."""
    return ast.literal_eval(
        evaluate(job["strategy"]["matrix"]["python-version"], github))


def push(sha):
    return {"workflow": "w", "event_name": "push", "ref": "refs/heads/main",
            "sha": sha, "event": {}}


def pull_request(sha):
    return {"workflow": "w", "event_name": "pull_request",
            "ref": "refs/pull/1/merge", "sha": sha,
            "event": {"pull_request": {"number": 1}}}


def schedule(sha, run_id):
    return {"workflow": "w", "event_name": "schedule", "sha": sha,
            "ref": "refs/heads/main", "run_id": run_id, "event": {}}


@pytest.fixture(
    params=["build", "benchmark", "lean", "paper", "proptest"])
def concurrency(request):
    with open(GITHUB / "workflows" / f"{request.param}.yml") as file:
        return yaml.safe_load(file)["concurrency"]


def test_every_push_to_main_has_a_group_of_its_own(concurrency):
    group = concurrency["group"]
    assert evaluate(group, push("a")) != evaluate(group, push("b"))
    assert evaluate(concurrency["cancel-in-progress"], push("a")) == "False"


def test_a_pull_request_supersedes_its_own_runs(concurrency):
    group = concurrency["group"]
    assert evaluate(group, pull_request("a")) == evaluate(
        group, pull_request("b"))
    assert evaluate(concurrency["cancel-in-progress"], pull_request("a"))\
        == "True"


def test_a_scheduled_proptest_run_never_cancels_a_push():
    """`proptest` is the only workflow on a schedule and a dispatch, and both
    fire on the commit a push already ran on."""
    with open(GITHUB / "workflows" / "proptest.yml") as file:
        group = yaml.safe_load(file)["concurrency"]["group"]
    assert evaluate(group, schedule("a", 1)) != evaluate(group, push("a"))
    assert evaluate(group, schedule("a", 1))\
        != evaluate(group, schedule("a", 2))


@pytest.fixture
def build_test():
    with open(GITHUB / "workflows" / "build.yml") as file:
        return yaml.safe_load(file)["jobs"]["test"]


def test_a_push_to_main_measures_every_python_of_its_matrix(build_test):
    """A job cancelled because a sibling failed is a version nothing
    measured, and `Test notebooks` runs on main and on no pull request: one
    red setup then leaves it run on none of them (#625). It costs nothing
    on a pull request, whose matrix is one version."""
    assert len(versions(build_test, push("a"))) > 1
    assert build_test["strategy"].get("fail-fast") is False
    assert versions(build_test, pull_request("a")) == ["3.14"]
