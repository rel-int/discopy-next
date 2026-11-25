> Review round on #737, cubic-dev-ai at 1249fbd and chatgpt-codex-connector at 7b8988c:
>
> > (Codex) Validate the wire count before gluing boundaries. (Fixed in 4bbc2a5.)
> > [P2] The top-level `feedback` import is now unused (test/comarkov.py).
> > [P2] This exclusivity claim is false: `pivotal.Ty` and `pivotal.Diagram` inherit from `feedback.Ty` and `feedback.Diagram` (ADR 0012).
> > [P3] This redirects every curated probe to `feedback.symmetric.Diagram`, leaving plain `discopy.symmetric.Diagram` without these rare-case corpus checks.

- [x] test/comarkov.py drop the unused feedback import
- [x] ADR 0012 the levels with a time step include pivotal and below
- [WIP] @amp-T-f829d881-2026-11-25 21:28 proptest/corpus/symmetric.py plain probes on symmetric, traced ones on feedback.symmetric
