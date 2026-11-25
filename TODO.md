> Review round on #737, cubic-dev-ai at 46ed830 and chatgpt-codex-connector at 2b3b7e7:
>
> > [P2] `glue` has no `n == 0` guard, unlike `trace`. (Codex: same finding.)
> > [P2] `GoISituation` promises a traced symmetric monoidal `bang`, but no axiom requires it to preserve `swap`.
> > [P2] `colaxator()` also represents the unit component, but these inverse axioms cover only `x, y`.
> > [P3] frobenius `Spider`/`Cospan` and closed `Pack` `__getnewargs__` docstrings still say a delay reads them.
> > [P3] paper/sections/syntax.typ: "It" after the `traced` sentence reads as the deprecated module.
> > [P3] interaction.py: the second local import is not covered by `disable-next`.
> > [P3] CHANGELOG Added entry still describes `discopy.bang`.
> > [P3] optyx/term.py `:mod:` references omit the `discopy.` prefix.

- [ ] discopy/hypergraph.py glue(0) is the identity
- [ ] discopy/abc.py GoISituation: bang preserves the swap
- [ ] discopy/abc.py GoISituation: unit colaxator inverse
- [ ] docstrings, paper, changelog, pylint comment that still say delay or bang
