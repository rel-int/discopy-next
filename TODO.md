make a new discopy-next branch (no PR) with all the upstream fixes we want to make. if not already the case, the constructors in `testable` should map to rules instantiations. feel free to make any other changes as well

- [x] `Constructor` subclasses `Rule`: the constructor `K` decorates its formation rule, `@K`, on the class of objects whose structure it is, a `Hom` of the objects' own category (ruled by joseph: on the types, not the categories)
- [x] formers on the abstract classes of objects: `ColouredMonoid` (`cut` = `Tensor`, `ax` = `Unit`, `form_repeat`), `ResiduatedMonoid` (`form_over`, `form_under`), `Pregroup` (`form_l`, `form_r`), new `DelayMonoid` (`form_delay`) and `BangMonoid` (`form_bang`); `feedback.Ty`, `bang.Ty`, `biclosed.Ty` take them
- [ ] an atom alphabet per category: abstract finite set, fixed list for semantics, `Nat`-indexed when infinite
- [ ] type equations as axioms: adjoint involution, `L = R` where it holds, delay over tensor, `Over` reducing to a tensor outside closed categories
- [WIP] @session_01C7YfAQS26HQesQ3UXcBUjA-2026-11-25 13:40 a strategy hook `Axiom.sequents` consults first, saying which strategy drew each law
- [x] `comarkov.Diagram.hypergraph_section` on `Trace(Merge(x, 2), left=True)`: fixed, `spider_factory` builds a spider with no leg out as a traced merge, the mirror of `markov`
- [ ] changelog entry, docs and tests for each change (the changelog bullets cite no pull request yet: add the `[next#N]` link once one is opened)
- [ ] every check of CONTRIBUTING.md's "Run the unit tests"
- [ ] report `comarkov.TermBase.str_transparency`, failing on `main` too: `str` of a comarkov term evaluates to a `monoidal.Kappa`
