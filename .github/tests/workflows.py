"""Tests for the workflows' concurrency groups.

A group keeps one running and one pending run, and a newly queued run
cancels the pending one whatever ``cancel-in-progress`` says: a workflow
that lets every commit on main keep its run needs a group per commit.
"""

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
        return str(eval(code, {"github": Context(github)}))
    return re.sub(r"\$\{\{(.*?)\}\}", value, str(expression))


def push(sha):
    return {"workflow": "w", "event_name": "push", "ref": "refs/heads/main",
            "sha": sha, "event": {}}


def pull_request(sha):
    return {"workflow": "w", "event_name": "pull_request",
            "ref": "refs/pull/1/merge", "sha": sha,
            "event": {"pull_request": {"number": 1}}}


@pytest.fixture(params=["build", "benchmark", "lean"])
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
