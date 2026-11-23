> > have a look at https://ampcode.com/threads/T-aaba85e4-d995-4e09-99ba-31fa238893e7 and come up with a strategy to reduce the proptest job to a workable size, that holds in the 30min CI time limit.
>
> > Yes, implement 1–4 in a PR and file both bugs as issues
>
> where 1–4 are, from the strategy proposed in https://ampcode.com/threads/T-6d395818-46db-4b9d-9787-884cf342289c (`main`'s proptest has timed out on every run since 2026-11-09, when the matrix grew from 248 to 2,908 cells: `explore` costs about 97× `pr` per cell, about 33 worker-hours in all):
>
> 1. Time-budgeted cells: the PR budget first, then fresh batches until a cell spends B seconds or reaches 1,000 examples.
> 2. `-n logical`: 4 workers instead of 2.
> 3. Pushes to `main`: one job, B ≈ 1 s, a fresh seed.
> 4. Nightly run: 4 jobs sharded by a hash of the cell name, B ≈ 5 s, then one job merging the reports and the example databases.

- [x] Time budget: a cell runs the pr budget, then fresh batches until PROPTEST_SECONDS or max_examples, and the report records the examples it ran
- [x] proptest workflow: -n logical, a push to main runs one job at B ~ 1 s with a fresh seed
- [x] proptest workflow: the nightly and a dispatch run 4 shards at B ~ 5 s, merged by one job into the report and the example database
- [ ] Docs: CONTRIBUTING.md's Run the property tests, and a changelog entry
