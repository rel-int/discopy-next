> chatgpt-codex-connector[bot], review of `107b7ac`, three P2 findings:
>
> > Account for a cut's rotation when evaluating its tensor (`discopy/peirce/alpha.py:191`)
>
> > Preserve the region colour when copying non-atomic boundaries (`discopy/peirce/beta.py:147`)
>
> > Default arbitrary relation boxes to non-cartesian flags (`discopy/peirce/beta.py:151`)

- [ ] a rotated cut evaluates to the rotation of its negated tensor
- [ ] & and | on empty and multi-wire boundaries: the empty type of a peirce diagram is white
- [ ] beta.Box is neither deterministic nor affine by default
