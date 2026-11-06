> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.

cubic's review of #456 at a79b3b6b:

- [x] **C1** pyproject.toml:110 (P3): The `test` extra now also bundles `notebooks`, but CONTRIBUTING.md still says it bundles only `quantum`, `grammar` and `optyx`. Add `notebooks` to that sentence so the doc matches the extra.
- [x] **C2** .github/tests/action.py:67 (P2): This test does not check that the setup action pins its actions. `pinned()` drops unpinned `uses:` entries, and `pins.get(name, {ref})` passes when no workflow uses that action. Assert that every `uses:` in the composite action carries a ref, and that any workflow pin for it matches.
- [ ] **C3** test/drawing/drawing.py:784 (P3): This parsing is brittle: a valid spelling with a space after the comma, like `discopy[grammar, notebooks]`, raises `KeyError` on `extras[" notebooks"]`. Strip each extra name, and compare requirement names rather than exact strings so `anywidget>=...` still matches.
