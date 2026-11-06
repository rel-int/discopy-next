# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Round of review feedback from `cubic`, quoted verbatim:

> **P2** (`discopy/stream.py:237`) This is not behavior-neutral as the PR claims. Because `Ty` and
> `Stream` are dataclasses that compare `_later`, `Ty.singleton(x) == Ty.singleton(x)` was False
> before and is now True. The `repr` of singleton streams also changes. Confirm that
> equality-by-value is intended for singletons, and cover it in a test or changelog. Otherwise keep
> identity-distinct thunks.

> **P3** (`CHANGELOG.md:699`) This overstates the lint state: `PLW0108` does not hold on `discopy/`
> yet, since Ruff 0.16.9 still reports one instance that this PR deliberately leaves for #443.
> Reword to say the two `singleton`s are fixed and one instance remains, or drop the claim.

Both are right and both are measured below. The P2 reading also turned up a wider defect, filed as
#457, which this pull request does not decide.

- [x] P3: reword the `CHANGELOG.md` entry to say two sites fixed, one instance left
- [x] P2: say in `CHANGELOG.md` that the two `singleton`s now compare equal to their twins
- [x] P2: a test in `test/stream.py` pinning that equality
- [x] P2: file the wider defect, every other non-constant stream comparing by closure identity
- [x] Re-run `ruff` at both versions, `pflake8`, `pylint`, `ty` and the suite
- [x] Reply on both threads and resolve them
