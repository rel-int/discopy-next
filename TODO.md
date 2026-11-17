> Review round on #591 (cubic-dev-ai[bot]), head 549cc08d:
>
> 1. .github/workflows/proptest.yml:19 (thread 4594068759)
> [P2] Per-SHA groups allow concurrent main runs, but this workflow still reads and writes shared, whole-snapshot artifacts without merging them. One run's Hypothesis discoveries and report can therefore be lost when the next run selects the other artifact; serialize artifact publication or merge per-run state before enabling concurrent exploration.
>
> 2. .github/workflows/proptest.yml:19 (thread 4594068775)
> [P2] The concurrency regression tests never exercise `proptest`, so this new group can regress to `github.ref` while the reported workflow tests still pass. Add `proptest` to the fixture parameters in `.github/tests/workflows.py`.
>
> 3. CHANGELOG.md:1589 (thread 4594068793)
> [P3] The changelog says `benchmark` groups pull requests by `github.ref`, but `.github/workflows/benchmark.yml` keys them by pull-request number. Describe this as a stable pull-request identifier instead.

- [x] .github/workflows/proptest.yml:19 Per-SHA groups allow concurrent main runs, but this workflow still reads and writes shared, whole-snapshot ar…, filed as #593
- [WIP] @session_01C7YfAN4j8VLLpH3mAacpDB-2026-11-17 01:39 .github/workflows/proptest.yml:19 The concurrency regression tests never exercise `proptest`, so this new group can regress to `github.ref` whi…
- [WIP] @session_01C7YfAN4j8VLLpH3mAacpDB-2026-11-17 01:39 CHANGELOG.md:1589 The changelog says `benchmark` groups pull requests by `github.ref`, but `.github/workflows/benchmark.yml` ke…
