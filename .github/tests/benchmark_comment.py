"""Tests for the benchmark comparison comment.

A ``workflow_run`` job runs with the privileges of the default branch on an
artifact staged by a run that has fewer, so most of these state what the
job refuses to post.
"""

import json

import pytest

BASE = "a" * 40
PREVIOUS = "b" * 40
HEAD = "c" * 40


@pytest.fixture
def data():
    return {"pull_request": 7, "base": BASE, "previous": PREVIOUS,
            "head": HEAD, "run_id": 99}


@pytest.fixture
def run():
    return {
        "id": 99, "head_sha": HEAD, "head_branch": "a-branch",
        "conclusion": "success", "html_url": "https://run",
        "head_repository": {"full_name": "discopy/discopy"},
        "pull_requests": [
            {"number": 7, "head": {"sha": HEAD}, "base": {"sha": BASE}}]}


@pytest.fixture
def pull():
    repo = {"full_name": "discopy/discopy", "html_url": "https://repo"}
    return {"head": {"sha": HEAD, "ref": "a-branch", "repo": repo},
            "base": {"sha": BASE, "repo": repo}}


def test_the_honest_metadata_passes(benchmark_comment, data, run, pull):
    assert benchmark_comment.unreadable(data, run) is None
    assert benchmark_comment.unattested(data, run, []) is None
    assert not benchmark_comment.retargeted(data, run, [])
    assert benchmark_comment.mismatch(
        data, run, pull, "discopy/discopy") is None


@pytest.mark.parametrize("field,value", [
    ("pull_request", "7"), ("pull_request", True), ("pull_request", None),
    ("run_id", 100), ("base", "z" * 40), ("previous", ""), ("head", "c" * 39),
    ("head", "d" * 40)])
def test_unreadable_rejects_a_forged_field(
        benchmark_comment, data, run, field, value):
    assert benchmark_comment.unreadable(
        dict(data, **{field: value}), run) == "Invalid benchmark metadata."


def test_mismatch_rejects_a_foreign_head_repository(
        benchmark_comment, data, run, pull):
    pull["head"]["repo"] = {"full_name": "someone/fork", "html_url": "x"}
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark metadata does not match its source PR.")


def test_mismatch_rejects_another_branch(
        benchmark_comment, data, run, pull):
    pull["head"]["ref"] = "another-branch"
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark metadata does not match its source PR.")


def test_unattested_rejects_a_run_belonging_elsewhere(
        benchmark_comment, data, run):
    run["pull_requests"] = [
        {"number": 8, "head": {"sha": HEAD}, "base": {"sha": BASE}}]
    assert benchmark_comment.unattested(data, run, []) == (
        "Benchmark run does not belong to this PR.")


def test_a_run_on_another_base_was_retargeted(
        benchmark_comment, data, run):
    """A run listing the pull request on another base still belongs to
    it: the pull request was retargeted while the run ran."""
    run["pull_requests"][0]["base"] = {"sha": "f" * 40}
    assert benchmark_comment.unattested(data, run, []) is None
    assert benchmark_comment.retargeted(data, run, [])


@pytest.mark.parametrize("fork", [False, True])
def test_main_skips_a_retargeted_run_with_a_notice(
        benchmark_comment, data, run, monkeypatch, tmp_path, capsys, fork):
    """A retargeted run is skipped, not failed: nothing is posted and the
    job exits 0, leaving the next run of the pull request to post. A run
    from a fork lists no pull request, and the open pull request for its
    head is the one on the new base."""
    listed = dict(run["pull_requests"][0], base={"sha": "f" * 40})
    run["pull_requests"] = [] if fork else [listed]
    if fork:
        run["head_repository"] = {"full_name": "someone/fork"}
    (tmp_path / "benchmark-comment").mkdir()
    (tmp_path / "benchmark-comment" / "metadata.json").write_text(
        json.dumps(data))
    (tmp_path / "event.json").write_text(json.dumps({"workflow_run": run}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "event.json"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "discopy/discopy")
    calls = []

    def call(url, *args):
        calls.append(url)
        assert "/pulls?head=" in url, f"{url} called for a retargeted run"
        return [listed]
    monkeypatch.setattr(benchmark_comment, "call", call)
    benchmark_comment.main()
    query = (f"{benchmark_comment.API}/repos/discopy/discopy/pulls"
             "?head=someone:a-branch&state=open")
    assert calls == ([query] if fork else [])
    assert capsys.readouterr().out == (
        "::notice::Skipping a comparison for a retargeted pull request.\n")


def test_unattested_names_the_one_open_pull_request_of_a_fork(
        benchmark_comment, data, run):
    """A run from a fork lists no pull request, so the metadata is held to
    the one open pull request for its head; a head carrying two of them
    names neither."""
    run["pull_requests"] = []
    assert benchmark_comment.unattested(data, run, [{"number": 7}]) is None
    for open_pulls in ([], [{"number": 8}], [{"number": 7}, {"number": 8}]):
        assert benchmark_comment.unattested(data, run, open_pulls) == (
            "Benchmark run does not name the one pull request for its head.")


def test_sanitised_neutralises_html_and_mentions(benchmark_comment):
    assert benchmark_comment.sanitised("<b> @toumix") == "&lt;b> &#64;toumix"


def test_body_links_the_merge_base(benchmark_comment, data, run, pull):
    body = benchmark_comment.body(data, run, pull, "a report")
    assert body.startswith(benchmark_comment.MARKER)
    assert "a report" in body
    assert f"merge base [`{PREVIOUS[:7]}`](https://repo/commit/{PREVIOUS})" \
        in body
    assert "[!WARNING]" not in body


def test_body_warns_when_the_base_moved(benchmark_comment, data, run, pull):
    pull["base"]["sha"] = "e" * 40
    run["conclusion"] = "failure"
    body = benchmark_comment.body(data, run, pull, "a report")
    assert "[!WARNING]" in body
    assert "base has changed" in body
    assert "concluded `failure`" in body


def test_ours_is_the_marked_comment(benchmark_comment):
    marker = benchmark_comment.MARKER
    comments = [
        {"id": 1, "user": {"login": "toumix"}, "body": f"{marker} a forgery"},
        {"id": 2, "user": {"login": "github-actions[bot]"}, "body": "hello"},
        {"id": 3, "user": {"login": "github-actions[bot]"},
         "body": f"{marker}\nthe comparison"}]
    assert benchmark_comment.ours(comments)["id"] == 3
    assert benchmark_comment.ours(comments[:2]) is None


def test_ours_takes_the_newest_of_several(benchmark_comment):
    """A duplicate must leave the stale comment behind, not the live one."""
    marker = benchmark_comment.MARKER
    bot = {"login": "github-actions[bot]"}
    comments = [
        {"id": 1, "user": bot, "body": marker, "created_at": "2026-08-01"},
        {"id": 9, "user": bot, "body": marker, "created_at": "2026-08-27"},
        {"id": 5, "user": bot, "body": marker, "created_at": "2026-08-10"}]
    assert benchmark_comment.ours(comments)["id"] == 9


def test_mismatch_rejects_a_deleted_head_repository(
        benchmark_comment, data, run, pull):
    pull["head"]["repo"] = None
    run["head_repository"] = None
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark metadata does not match its source PR.")
