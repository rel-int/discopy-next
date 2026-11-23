> Review round 1 of #702, cubic-dev-ai at c32686fb (https://github.com/rel-int/discopy-next/pull/702#pullrequestreview-6022010736):
>
> > 
> > 1. proptest/report.py:308 [cubic-dev-ai[bot]] thread-root-id=4666065220
> > [P2] `merge` accepts the same cell name from two slice reports without complaint. The merged result then depends on argument order: the last report wins in `stopped` and `dumps`, and `render` double-counts the cell. Reject duplicate names so an overlapping or misassigned shard fails loudly.
> > 
> > ```suggestion
> >     found = [cell for report in reports for cell in loads(report)]
> >     names = [cell.name for cell in found]
> >     if len(set(names)) != len(names):
> >         raise ValueError("a cell appears in more than one slice report")
> >     cells = tuple(sorted(found, key=lambda cell: cell.name))
> > ```
> > 
> > 2. .github/tests/workflows.py:133 [cubic-dev-ai[bot]] thread-root-id=4666065246
> > [P2] This test claims to cover dispatch runs but never evaluates a `workflow_dispatch` event. Add a dispatch context and assert it also expands to `[0, 1, 2, 3]`, otherwise the dispatch-specific matrix branch is unprotected.
> > 
> > 3. .github/workflows/proptest.yml:102 [cubic-dev-ai[bot]] thread-root-id=4666065254
> > [P2] `PROPTEST_STOP` is computed when the run step starts, so checkout, setup and the downloads before it are not counted against the 22 minutes. The deadline can land 25 minutes or more into a 30-minute job, and a timeout would skip the report and database uploads. Compute the deadline in a first step of the job so it is measured from the job start, e.g. `echo "PROPTEST_STOP=$(( $(date +%s) + 22 * 60 ))" >> "$GITHUB_ENV"`, and drop the inline assignment from the run command.
> > 
> > 4. proptest/conftest.py:236 [cubic-dev-ai[bot]] thread-root-id=4666065265
> > [P2] Validate the shard index and count before slicing; otherwise malformed values can crash collection or make CI report success without running any tests. Raise `pytest.UsageError` unless `0 <= index < count` and `count > 0`.
> > 
> > ```suggestion
> >     try:
> >         index, count = map(int, option.split("/"))
> >     except ValueError as error:
> >         raise pytest.UsageError("--proptest-shard must be I/N") from error
> >     if count <= 0 or not 0 <= index < count:
> >         raise pytest.UsageError(
> >             "--proptest-shard must have 0 <= I < N and N > 0")
> > ```
> > 
> > 5. proptest/test_conftest.py:49 [cubic-dev-ai[bot]] thread-root-id=4666065282
> > [P2] This assertion relies on random generation covering both Boolean values in the first batch, so the test can be flaky. Assert the two-batch finite-support stop without requiring both values, or make the generated sequence deterministic.
> > 
> > ```suggestion
> >     assert drawn <= {False, True} and len(sizes) == 2
> > ```
> > 
> > 6. proptest/test_conftest.py:48 [cubic-dev-ai[bot]] thread-root-id=4666065297
> > [P2] This unit test can execute nearly one million Hypothesis examples in its second batch. Use a small cap larger than `FIRST` so the finite-support stop is tested without making the suite unnecessarily slow.
> > 
> > ```suggestion
> >     sizes, drawn = _sizes(st.booleans(), seconds=600, cap=100)
> > ```
> > 
> > 7. CONTRIBUTING.md:261 [cubic-dev-ai[bot]] thread-root-id=4666065304
> > [P2] `PROPTEST_STOP` is only checked after the current Hypothesis batch returns, so it cannot guarantee this outcome. A slow first 20-example batch can reach the 30-minute job timeout before later cells fall back to pull-request coverage; qualify this as best-effort or enforce a per-batch deadline.
> > 
> > ```suggestion
> > a run that outgrows its 30 is intended to fall back to what a pull request checks,
> > though a slow first batch can still reach the job timeout. It carries the report
> > ```
> > 
> > 8. CONTRIBUTING.md:257 [cubic-dev-ai[bot]] thread-root-id=4666065316
> > [P3] This paragraph recommends `-n logical`, but the same file still says `-n auto` runs on all cores and uses it in the local commands. Either explain in one place why CI uses `logical` (`auto` counts physical cores) or align the worker flag so readers do not get two different recommendations.
> > 
> > 9. CONTRIBUTING.md:240 [cubic-dev-ai[bot]] thread-root-id=4666065329
> > [P3] The batch sizing is half the time left, not the time left, as `batches()` in `proptest/conftest.py` computes it. Say "half the time left" so the doc matches the code.
> > 
> > ```suggestion
> > of a pull request first, then batches of fresh ones sized to half the time left,
> > ```
> > 
> > 10. CHANGELOG.md:1971 [cubic-dev-ai[bot]] thread-root-id=4666065338
> > [P3] This entry omits manual dispatches, although the workflow gives them the same four five-second shards and merge job as nightly runs. Mention dispatched runs alongside nightly runs so users know manual deep searches use the new configuration.
> > 
> > 11. .github/workflows/proptest.yml:103 [cubic-dev-ai[bot]] thread-root-id=4666065346
> > [P3] Pull-request runs now use `-n logical` instead of the previous `-n auto`, so their worker count and resource usage change despite the stated requirement that pull-request runs remain unchanged. Select `auto` for `pull_request` and `logical` only for the deeper searches.

- [WIP] @T-6d395818-2026-11-23 12:53 proptest/report.py merge: reject a cell named by two slices (r4666065220)
- [WIP] @T-6d395818-2026-11-23 12:53 proptest/conftest.py: --proptest-shard rejects a malformed or out-of-range I/N (r4666065265)
- [WIP] @T-6d395818-2026-11-23 12:53 proptest/test_conftest.py: finite-support test with a small cap, not relying on both booleans (r4666065282, r4666065297)
- [WIP] @T-6d395818-2026-11-23 12:53 proptest.yml: PROPTEST_STOP from the first step of the job; workflows test covers a dispatch (r4666065254, r4666065246)
- [WIP] @T-6d395818-2026-11-23 12:53 Docs: half the time left, a best-effort deadline, one worker flag, dispatch in the changelog (r4666065304, r4666065316, r4666065329, r4666065338)
- [WIP] @T-6d395818-2026-11-23 12:53 Decline -n auto on pull requests: -n logical everywhere is item 2 of the approved plan (r4666065346)
