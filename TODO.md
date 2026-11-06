cubic-dev-ai's review of 920624d2, 11-06:

> **9 issues found** across 41 files
>
> - `test/config.py:32` P2: `label_sizes` discards duplicate labels, so this test cannot detect a missing or incorrectly sized second `f` box label in `Equation(f, f)`. Preserve multiplicity with a list or `Counter` and compare counts in the expected result.
> - `.github/scripts/benchmark_comment.py:135` P2: `retargeted()` only scans `run["pull_requests"]`, but fork runs have no entries; after `unattested()` accepts the sole `open_pulls` PR, a retargeted fork run is not skipped and the old-base comparison is posted. Include `open_pulls` or the fetched PR base in this check.
> - `test/fixtures/pyproject.py:19` P2: `installs(GROUPS, "all")` fails because the `all` group contains `{ "include-group": ... }` dictionaries, which this loop passes to `Requirement`. Handle included groups before parsing string requirements so the helper supports every exported dependency group.
> - `discopy/kleisli/monad.py:567` P2: Equal canonical keys preserve hash-dependent `frozenset` order, so the same sampling seed can select different outcomes across hash seeds despite this new determinism guarantee. Reject ambiguous canonical keys or require/provide a deterministic tie-breaker for outcomes with equal canonical representations.
> - `CHANGELOG.md:258` P2: This determinism claim is too broad: equal canonical strings preserve hash-dependent iteration order. Restrict the note to values with unique canonical representations, or document the limitation.
> - `CHANGELOG.md:262` P3: Correct the spelling of `braket` to `bracket` in the drawing changelog entry.
> - `discopy/drawing/backend.py:1478` P2: Raw xcolor expressions containing commas become part of the TikZ style key here, but `style=...` parses commas as option separators. Encode non-key characters (or use a deterministic digest) before embedding arbitrary colors in the generated style name.
> - `.github/workflows/proptest.yml:39` P3: `HYPOTHESIS_GITHUB_REPOSITORY` is unused in every workflow run: this job always selects `pr` or `explore`, while `conftest.py` reads it only for `shared`. Remove this dead job-level setting and its comments.
> - `discopy/quantum/gates.py:77` P2: `exact=True` still loses phase values by coercing them to `float`, so `eval(repr(g)) == g` fails for large integers and exact numeric types such as `Fraction` or `Decimal`; preserve the original value in the representation or normalize phases explicitly before comparison.

Each finding is checked against the code first: a real one is fixed with a test that fails on 920624d2, and the others are answered on their thread.

- [ ] S9-9: `label_sizes` keeps the multiplicity of labels
- [ ] S16-3: a retargeted run is skipped whether or not it lists its pull request
- [ ] S17-9: `installs` reads an `include-group` entry
- [ ] S8-13: `sample` breaks ties between equal canonical keys, and the changelog says what it guarantees
- [ ] S9-9: "braket" is the drawing's word for bras and kets: answered, not changed
- [ ] S9-17: a TikZ style name holds no character that ends a key
- [ ] S2-11: the proptest job sets no variable that no profile it selects reads
- [ ] S10-2: the exact repr of a phase reads back to an equal phase
