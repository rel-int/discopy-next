> chatgpt-codex-connector[bot], review of `daf82a2`, two P2 findings:
>
> > Preserve declared box flags through photographic negatives (`discopy/peirce/alpha.py:119`)
>
> > Register beta-specific cup and cap factories (`discopy/peirce/beta.py:254`)

- [x] negative() rebuilds each box with all its attributes, so it is an involution on flagged boxes
- [x] beta.Cup and beta.Cap, so cups, caps and traces build in beta
