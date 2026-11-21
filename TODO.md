> yes let's start a thread for that architecture change, I think bang should go everywhere the same way feedback goes everywhere, the user can always decide not to use it

toumix, answering his own question "where does bang fit in the hierarchy? every monoidal category could use it?". His rulings on the design ([#686](https://github.com/rel-int/discopy-next/pull/686#issuecomment-7062218939), 2026-11-21): build it as proposed; at `markov` the free comonad, with `Contraction` as `Copy(!x)`, `Weakening` as `Discard(!x)` and `Bang` boxes deterministic; its own PR, stacked on `amp/bang-module`. The points of #686's own TODO.md stay on #686.

- [x] `bang.structural_wires` and `bang.structural_boxes` generate the `BangOb` and the structural boxes of a level, the way `feedback.structural_boxes` does; the bang terms come through `cat.kappa_calculus`, whose parents are now bang terms
- [x] `bang.Diagram` a base of `spacial`, `braided`, `feedback` and `biclosed`, and `bang.Ty` a base of their types, so that every level above inherits a bang
- [x] `Functor` of each level sends banged wires and boxes through the bang of its codomain, or through `ob_map` and `ar_map` when it has none
- [x] from `symmetric` up `Diagram.exchange` is the swap and the bang terms wire by a permutation
- [x] at `markov` `Contraction(x)` is `Copy(x.bang())`, `Weakening(x)` is `Discard(x.bang())`, `Bang` is deterministic, and the bang terms are the markov terms
- [x] a `Ty` of their own for `spacial`, `braided`, `symmetric`, `markov` and `comarkov` with a bang, once #679 merges: #679 is merged into this branch
- [ ] the bang laws of each level in the property matrix, declared failing or holding
- [ ] `optyx.Term`, its tests and the `optyx/linear_logic` notebook unchanged and passing
- [x] docs, CHANGELOG bullet and an ADR for where the bang lives
- [ ] a banged wire has no adjoint: from `rigid` up `x.bang().l` raises; file it as an issue or give `BangOb` a winding
- [ ] enrol bang.TermBase and the bang terms of each level in the property matrix (moved from #686)
