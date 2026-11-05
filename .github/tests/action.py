"""Tests for the composite action the build jobs share.

A composite step takes a strict subset of what a workflow step takes, and
the runner only says so when the job starts: `timeout-minutes` on one of
these cost a whole CI round.
"""

import fnmatch
import pathlib

import pytest
import yaml

GITHUB = pathlib.Path(__file__).resolve().parent.parent
STEP = {"name", "id", "if", "uses", "with", "run", "shell", "env",
        "working-directory", "continue-on-error"}


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
    for name, ref in pinned(action["runs"]["steps"]):
        assert pins.get(name, {ref}) == {ref}, name


def test_dependabot_updates_the_setup_action():
    with open(GITHUB / "dependabot.yml") as file:
        config = yaml.safe_load(file)
    actions, = (update for update in config["updates"]
                if update["package-ecosystem"] == "github-actions")
    directories = actions.get("directories", [actions.get("directory")])
    assert any(fnmatch.fnmatch("/.github/actions/setup", pattern)
               for pattern in directories)
