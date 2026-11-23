> I think we're good to go with the Term implementation throughout all modules, tell the discopy-lean space to start formalising a subset of it for a test.

- [ ] cat: nameless terms of one variable, beta normal form, evaluation into any category, and the five laws of cat.TermBase (eta, normal_form_soundness, normal_form_idempotence, normal_form_completeness, reading)
- [ ] monoidal: planar linear terms indexed by their context, evaluation into Diagram(S), eta, and reading a diagram with one output as a term that evaluates back to it
- [ ] Statements: restate the definitions and theorems in DisCoLean/Statements/Terms.lean, audited by Audit.lean
- [ ] Docs: discolean/README.md, CONTRIBUTING.md's Lean section and a changelog entry
