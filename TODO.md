> yes let's start a thread for that architecture change, I think bang should go everywhere the same way feedback goes everywhere, the user can always decide not to use it

toumix, answering his own question "where does bang fit in the hierarchy? every monoidal category could use it?". His rulings on the design ([#686](https://github.com/rel-int/discopy-next/pull/686#issuecomment-7062218939), 2026-11-21): build it as proposed; at `markov` the free comonad, with `Contraction` as `Copy(!x)`, `Weakening` as `Discard(!x)` and `Bang` boxes deterministic; its own PR, stacked on `amp/bang-module`. The points of #686's own TODO.md stay on #686.

- [WIP] @T-6ea8260d-2026-11-21 12:13 `bang.structural_wires`, `bang.structural_boxes` and `bang.linear_calculus` generate the `BangOb`, the structural boxes and the bang terms of a level, the way `feedback.structural_boxes` and `cat.kappa_calculus` do
- [ ] `bang.Diagram` a base of `spacial`, `braided`, `feedback` and `biclosed`, and `bang.Ty` a base of their types, so that every level above inherits a bang
- [ ] `Functor` of each level sends banged wires and boxes through the bang of its codomain, or through `ob_map` and `ar_map` when it has none
- [ ] from `symmetric` up `Diagram.exchange` is the swap and the bang terms wire by a permutation
- [ ] at `markov` `Contraction(x)` is `Copy(x.bang())`, `Weakening(x)` is `Discard(x.bang())`, `Bang` is deterministic, and the bang terms are the markov terms
- [ ] a `Ty` of their own for `spacial`, `braided`, `symmetric`, `markov` and `comarkov` with a bang, once #679 merges
- [ ] the bang laws of each level in the property matrix, declared failing or holding
- [ ] `optyx.Term`, its tests and the `optyx/linear_logic` notebook unchanged and passing
- [ ] docs, CHANGELOG bullet and an ADR for where the bang lives
