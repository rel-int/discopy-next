make a new discopy-next branch (no PR) with all the upstream fixes we want to make. if not already the case, the constructors in `testable` should map to rules instantiations. feel free to make any other changes as well

- [WIP] @session_01C7YfAQS26HQesQ3UXcBUjA-2026-11-25 10:00 `Constructor` subclasses `Rule`: one `Var` premise per type parameter, conclusion `Ob[C0, ...]`, `form` the implementation
- [ ] constructors declared on the abstract base classes and inherited (monoidal: `Tensor`, `Unit`; rigid: `L`, `R`; feedback: `Delay`; closed: `Over`, `Under`; ...)
- [ ] an atom alphabet per category: abstract finite set, fixed list for semantics, `Nat`-indexed when infinite
- [ ] type equations as axioms: adjoint involution, `L = R` where it holds, delay over tensor, `Over` reducing to a tensor outside closed categories
- [ ] a strategy hook `Axiom.sequents` consults first, saying which strategy drew each law
- [ ] `comarkov.Diagram.hypergraph_section` on `Trace(Merge(x, 2), left=True)` (raises: the only spiders of comarkov.Diagram are its merges): fix if easy, else declare failing
- [ ] changelog entry, docs and tests for each change
- [ ] every check of CONTRIBUTING.md's "Run the unit tests"
