make a new discopy-next branch (no PR) with all the upstream fixes we want to make. if not already the case, the constructors in `testable` should map to rules instantiations. feel free to make any other changes as well

- [x] `Constructor` subclasses `Rule`: the constructor `K` decorates its formation rule, `@K`, on the class of objects whose structure it is, a `Hom` of the objects' own category (ruled by joseph: on the types, not the categories)
- [x] formers on the abstract classes of objects: `ColouredMonoid` (`cut` = `Tensor`, `ax` = `Unit`, `form_repeat`), `ResiduatedMonoid` (`form_over`, `form_under`), `Pregroup` (`form_l`, `form_r`), new `DelayMonoid` (`form_delay`) and `BangMonoid` (`form_bang`); `feedback.Ty`, `bang.Ty`, `biclosed.Ty` take them
- [WIP] @session_01C7YfAQS26HQesQ3UXcBUjA-2026-11-25 14:10 an atom alphabet per category: abstract finite set, fixed list for semantics, `Nat`-indexed when infinite
- [x] type equations as axioms on the classes of types: `Pregroup` (adjoint involution, contravariance, residuation), `DelayMonoid` (delay preserves tensor and unit); `L = R` is `PivotalCategory.self_dual` already
- [x] a strategy hook: `testable.PROVIDERS`, tried by `Declaration.sequents` before its own draw; `Declaration.provider` names it, noted in each matrix cell
- [x] `comarkov.Diagram.hypergraph_section` on `Trace(Merge(x, 2), left=True)`: fixed, `spider_factory` builds a spider with no leg out as a traced merge, the mirror of `markov`
- [ ] changelog entry, docs and tests for each change (the changelog bullets cite no pull request yet: add the `[next#N]` link once one is opened)
- [ ] every check of CONTRIBUTING.md's "Run the unit tests"
- [ ] report `comarkov.TermBase.str_transparency`, failing on `main` too: `str` of a comarkov term evaluates to a `monoidal.Kappa`
