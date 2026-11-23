"""
Unit tests for the profiles of :mod:`proptest.conftest`: which repository's
example database the ``shared`` profile replays, and how a cell spends a
time budget.
"""

from hypothesis import Phase
from hypothesis import strategies as st
from hypothesis.database import GitHubArtifactDatabase

from proptest.conftest import FIRST, artifact, batches


def test_shared_replays_upstream_by_default(monkeypatch):
    """ Unset, the variable leaves the file as it is upstream. """
    monkeypatch.delenv("HYPOTHESIS_GITHUB_REPOSITORY", raising=False)
    assert artifact() == GitHubArtifactDatabase("discopy", "discopy")


def test_shared_replays_the_repository_the_environment_names(monkeypatch):
    """ As this repo's CI and CONTRIBUTING.md set it. """
    monkeypatch.setenv("HYPOTHESIS_GITHUB_REPOSITORY", "rel-int/discopy-next")
    assert artifact() == GitHubArtifactDatabase("rel-int", "discopy-next")


def _sizes(strategy, **budget):
    """ The batches a body drawing from ``strategy`` runs, and its terms. """
    drawn = set()
    sizes = list(batches(
        lambda data: drawn.add(data.draw(strategy)),
        tuple(Phase), drawn, **budget))
    return sizes, drawn


def test_without_seconds_a_cell_runs_its_examples_at_once():
    """ As a pull request does, whatever the strategy. """
    assert _sizes(st.integers(), seconds=0, cap=37)[0] == [37]


def test_a_budget_runs_past_the_first_batch_up_to_the_cap_exactly():
    """ The batches add up to the cap, never past it. """
    sizes, _ = _sizes(st.integers(), seconds=600, cap=77)
    assert sizes[0] == FIRST and len(sizes) > 1 and sum(sizes) == 77


def test_a_finite_support_stops_a_budget_after_a_batch_draws_nothing_new():
    """
    Rather than draw the same two terms until its ten minutes are up: the
    cap is out of reach, so only the support stops it, and costs nothing,
    since Hypothesis ends a search that has exhausted its space.
    """
    sizes, drawn = _sizes(st.booleans(), seconds=600, cap=10 ** 6)
    assert drawn <= {False, True} and len(sizes) == 2


def test_past_the_stop_a_cell_runs_its_first_batch_only():
    """ What a pull request checks, when the job has no time left. """
    assert _sizes(st.integers(), seconds=600, stop=0, cap=1000)[0] == [FIRST]
