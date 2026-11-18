> chatgpt-codex-connector[bot], review of `107b7ac`, three P2 findings:
>
> > Account for a cut's rotation when evaluating its tensor (`discopy/peirce/alpha.py:191`)
>
> > Preserve the region colour when copying non-atomic boundaries (`discopy/peirce/beta.py:147`)
>
> > Default arbitrary relation boxes to non-cartesian flags (`discopy/peirce/beta.py:151`)

- [x] a rotated cut evaluates to the rotation of its negated tensor
- [x] & and | on empty and multi-wire boundaries: the empty type of a peirce diagram is white
- [x] beta.Box is neither deterministic nor affine by default
- [x] negative() keeps each wire's z (cubic P1)
- [WIP] @session_01MEJxy4BLudMDJBi1HBa5XG-2026-11-18 17:16 Model compares its maps by content and is not hashable (cubic P2, P3)
- [WIP] @session_01MEJxy4BLudMDJBi1HBa5XG-2026-11-18 17:16 tests: no tautology, relative_sum and par truth values, axioms on several models; alpha.Box example on the unit; axiom docs; CHANGELOG cites in prose (cubic)
