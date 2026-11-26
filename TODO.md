> Alexis decided to split `discopy.closed` into a package. Open one pull request on rel-int/discopy-next for it. Do the work in this thread; do not start another thread.
>
> His words: "compact is not a subclass of closed which is a mistake due to fear of boilerplate, we should fix it. that means we can't have closed be markov by default, we could have a whole subpackage with closed.monoidal (current biclosed), closed.symmetric and closed.markov" and "let's do the closed split in a first PR".
>
> What that means:
> - `discopy.closed` becomes a package. `closed.monoidal` is the current `discopy.biclosed`. `closed.symmetric` is new: symmetric monoidal closed, not Markov. `closed.markov` is the current `discopy/closed.py` (copy, discard, exponentials, its terms with tuples, projections and `let`, Böhm trees).
> - `compact` subclasses `closed.symmetric`, with the internal hom x ⊸ y = x* ⊗ y, so eval and curry come from cups and caps. Its cup-and-cap trace stays as it is.
> - Not decided, so do not do it: `rigid` and `pivotal` under `closed.monoidal`. If it turns out to fall out with no extra work, say so in the PR instead.
> - Not in this PR: anything traced. `feedback.closed` stays feedback over the Markov case and is repointed to `closed.markov`. Do not add a traced `closed.symmetric`; that is a later PR.
> - Keep `closed` and `feedback` one-way: `feedback` may import `closed`, `closed` never imports `feedback`, and nothing here imports `discopy.linear`.
>
> Base: this builds on #737 (feedback and linear as packages), which is not merged yet and moves `feedback/closed.py`. Branch from #737's branch `amp/feedback-bang-packages` (head b9a2073 when this was written) and open the PR against that branch, saying in the description that it is stacked on #737. If #737 merges first, retarget to `main`.
>
> Decide yourself whether the old `discopy.biclosed` and the flat `discopy.closed` names keep deprecated aliases for one release, the way `discopy/traced.py` did, and say which in the PR.

- [WIP] @T-01a21749-9871-73a9-9e3f-e31cfdd7435a-2026-11-26 10:24 `discopy.closed` is a package: `closed.monoidal` (was `biclosed`), a new `closed.symmetric`, `closed.markov` (was `closed`)
- [WIP] @T-01a21749-9871-73a9-9e3f-e31cfdd7435a-2026-11-26 10:24 `compact` subclasses `closed.symmetric`, eval and curry from cups and caps
- [WIP] @T-01a21749-9871-73a9-9e3f-e31cfdd7435a-2026-11-26 10:24 `feedback.closed` repointed to `closed.markov`
- [ ] deprecated aliases for `discopy.biclosed` and the flat `discopy.closed` names, for one release
- [ ] docs, paper, changelog and an architecture decision record
