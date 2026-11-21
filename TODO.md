> Set up architecture decision records (ADRs) in discopy-next. Do the work yourself; don't spawn another thread.
>
> 1. Read RULES.md, STYLE.md, CONTRIBUTING.md and CHANGELOG.md first. If RULES.md already records design decisions, have the ADRs take that part over rather than duplicate it, and say what moves.
> 2. Propose a short format under docs/adr/ (NNNN-title.md, Context / Decision / Consequences / Status, under a page each, with a Superseded-by convention). Add a brief note in CONTRIBUTING.md: one ADR per design decision, in the same PR as the change. The changelog stays as is.
> 3. Seed five or six ADRs from these decisions (check each against the code and PRs, don't trust this summary):
>    - discopy.grammar → discopy.lambek with a deprecated alias (#569); discopy.axioms → discopy.testable with no alias (#403)
>    - A wire's equality counts is_dagger, and so does cat.Bubble's setoid, against upstream's commuting law (#507, #498)
>    - stream.Stream equality is intensional (#535)
>    - Bubble fuses into Box (#560), plumbing classes share #513's functor rule (#561)
>    - A fast specialised implementation lands beside the slow general one that validates it (e.g. interaction nets beside Hypergraph.rewrite)
>    - The tool paper is LaTeX only, reversing the earlier Typst decision (the library keeps its Typst backend); record it as an ADR that supersedes the earlier one
>
> Keep it small to review. When you're done, state the review cost (lines changed in existing files, lines in new files) and reply to this thread with the result.

- [x] docs/adr/: the format, an index in the docs
- [x] CONTRIBUTING.md: one ADR per design decision, in the same PR
- [x] Seed ADRs, each checked against the code and the PRs
