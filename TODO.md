> spawn another two to shard proptest and to answer the leftover decisions

USER picked option 1 of #486 (shard `explore` over runners, keep the budget).

- [WIP] @session_01MEJxy5ajMSC6icUps5JQd8-2026-11-09 shard the `proptest` workflow: `--proptest-shard=I/N`, a matrix of shards on `main`, one on a pull request
- [ ] merge the shards' reports into the one the baseline compares against and uploads
- [ ] merge the shards' example databases without bringing back what a shard pruned
- [ ] tests for the split and both merges, CONTRIBUTING.md, CHANGELOG.md
- [ ] dispatch `explore` on the branch, report the shards' durations
