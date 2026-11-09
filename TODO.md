> spawn another two to shard proptest and to answer the leftover decisions

USER picked option 1 of #486 (shard `explore` over runners, keep the budget).

- [x] shard the `proptest` workflow: `--proptest-shard=I/N`, a matrix of shards on `main`, one on a pull request
- [x] merge the shards' reports into the one the baseline compares against and uploads
- [x] merge the shards' example databases without bringing back what a shard pruned
- [x] tests for the split and both merges, CONTRIBUTING.md, CHANGELOG.md
- [x] dispatch `explore` on the branch: 20 shards at 30 min left 9 cancelled, 40 at 30 left 2, 40 at 60 all done (median 13.5 min, max 56, on #491's two cells)
- [ ] USER: per-shard timeout, see the 🚀 comment
