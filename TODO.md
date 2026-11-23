> toumix, in the Amp thread of this pull request (https://ampcode.com/threads/T-0729b388-26a3-4a3d-ac11-e727fdb06d74), on the three points it left open (#683, #684 and the paper draft on rel-int/wiki#195):
>
> > 1) yes let's change ribbon and compact too: every module its own Ty
> > 2) can we find a way to raise the error at construction time instead?
> > 3) edit the draft accordingly

- [x] ribbon and compact each get a Ty and a Nat of their own, subclassing those of the levels their diagrams subclass; frobenius.Ty is a compact.Ty (#683)
- [WIP] @T-0729b388-26a3-4a3d-ac11-e727fdb06d74-2026-11-23 17:15 a kappa over a constant of another level raises when it is built, stated once for every level (#684)
- [ ] tests, the trees and pickles written with pivotal.Ty, docs/adr/0008 and the changelog
- [ ] the paper draft on rel-int/wiki#195 follows: terms of each module, no discopy.kappa
