> I set up a new environment for discopy-lean, can you spawn a fresh session to prove the statement "discopy diagrams are the morphisms of the free premonoidal category"?

- [x] Slice 1, the category: define category and strict premonoidal category in lean/Premonoidal, Diagram(Σ) with id, then, whiskering and tensor
- [ ] Prove premonoidal, tensor_def and not_monoidal, restated in lean/Statements/Premonoidal.lean and covered by Audit.lean
- [ ] The oracle answers then, tensor and whiskering on encoded diagrams; lean/test_premonoidal.py agrees with monoidal.Diagram on 50,000 cases over three seeds, and one seeded mutant fails it
- [ ] Docs: lean/README.md, CONTRIBUTING.md, CHANGELOG.md, the lean workflow's paths
