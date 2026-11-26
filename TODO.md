# TODO

> Follow ~/workspace/repo/EVENING.md

Part of #731, taken under *Bugs in a next repo*: the issue states the fix, a `to_tree`/`from_tree`
of each box's own constructor arguments, as `Feedback` has.

- [x] `channel.Discard`, `Measure` and `Encode` write their `dom` (and internal states) and read it back, and their `n`-subclasses in `photonic` and `qubits` read `n` from it
- [x] `photonic.Phase`, `BBS`, `TBS` and `MZI` write their parameters and flags and read them back
- [x] test: `loads(dumps(box)) == box` on each
- [x] CHANGELOG bullet with the PR number
- [x] the gates of next-checks: pflake8, pylint, ruff, ty, the optyx tests
