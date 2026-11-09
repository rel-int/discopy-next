> spawn another two to shard proptest and to answer the leftover decisions

USER picked option 1 of #486 (shard `explore` over runners, keep the budget).

- [x] shard the `proptest` workflow: `--proptest-shard=I/N`, a matrix of shards on `main`, one on a pull request
- [x] merge the shards' reports into the one the baseline compares against and uploads
- [x] merge the shards' example databases without bringing back what a shard pruned
- [x] tests for the split and both merges, CONTRIBUTING.md, CHANGELOG.md
- [WIP] @session_01MEJxy5ajMSC6icUps5JQd8-2026-11-09 21:20 dispatch `explore` on the branch, report the shards' durations: 20 shards left 9 at 30 min, 40 left 2, both on one cell of `interaction` over 15 min
- [ ] USER: per-shard timeout, see the 🚀 comment
