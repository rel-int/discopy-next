"""Tests for the composite action the build jobs share.

A composite step takes a strict subset of what a workflow step takes, and
the runner only says so when the job starts: `timeout-minutes` on one of
these cost a whole CI round.
"""

import fnmatch
import pathlib
import re

import pytest
import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent
STEP = {"name", "id", "if", "uses", "with", "run", "shell", "env",
        "working-directory", "continue-on-error"}
SHA = re.compile(r"[0-9a-f]{40}")


@pytest.fixture(scope="module")
def action():
    with open(GITHUB / "actions" / "setup" / "action.yml") as file:
        return yaml.safe_load(file)


@pytest.fixture(scope="module")
def workflow_steps():
    steps = []
    for path in (GITHUB / "workflows").glob("*.yml"):
        with open(path) as file:
            workflow = yaml.safe_load(file)
        for job in workflow["jobs"].values():
            steps += job.get("steps", [])
    return steps


def pinned(steps):
    for step in steps:
        name, at, ref = step.get("uses", "").partition("@")
        if at:
            yield name, ref


def unpinned(steps, pins):
    """
    The ``uses:`` of the steps whose ref is not a commit SHA, or not the one
    ``pins`` holds for that action. A local action is read from the
    workspace and has no ref.
    """
    for step in steps:
        uses = step.get("uses")
        if uses is None or uses.startswith("./"):
            continue
        name, _, ref = uses.partition("@")
        if not SHA.fullmatch(ref) or pins.get(name, {ref}) != {ref}:
            yield uses


def test_every_step_key_is_one_a_composite_step_takes(action):
    for step in action["runs"]["steps"]:
        assert set(step) <= STEP, f"{set(step) - STEP} in {step.get('name')}"


def test_every_script_names_its_shell(action):
    for step in action["runs"]["steps"]:
        assert "run" not in step or step.get("shell"), step.get("name")


def test_every_input_the_workflows_pass_is_declared(action, workflow_steps):
    passed = set()
    for step in workflow_steps:
        if step.get("uses", "").startswith("./.github/actions/setup"):
            passed |= set(step.get("with", {}))
    assert passed and passed <= set(action["inputs"])


def test_every_action_the_setup_uses_is_pinned_as_in_the_workflows(
        action, workflow_steps):
    pins = {}
    for name, ref in pinned(workflow_steps):
        pins.setdefault(name, set()).add(ref)
    assert not list(unpinned(action["runs"]["steps"], pins))


def test_a_tag_or_another_sha_than_the_workflows_is_unpinned():
    sha, other = "0" * 40, "1" * 40
    steps = [{"uses": "a/tag@v1"}, {"uses": "a/bare"},
             {"uses": f"a/new@{sha}"}, {"uses": f"a/moved@{sha}"},
             {"uses": "./local"}, {"run": "true"}]
    assert list(unpinned(steps, {"a/moved": {other}})) == [
        "a/tag@v1", "a/bare", f"a/moved@{sha}"]


def test_dependabot_updates_the_setup_action():
    with open(GITHUB / "dependabot.yml") as file:
        config = yaml.safe_load(file)
    actions, = (update for update in config["updates"]
                if update["package-ecosystem"] == "github-actions")
    directories = actions.get("directories", [actions.get("directory")])
    assert any(fnmatch.fnmatch("/.github/actions/setup", pattern)
               for pattern in directories)
