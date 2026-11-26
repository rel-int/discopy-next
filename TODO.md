> Alexis (toumix), relayed from https://ampcode.com/threads/T-01a21405-6d38-764f-a655-35b13cc7bcb4:
>
> > I think we finally got round to a convincing hierarchy, write it that way and start a thread that will begin with an updated version of the architecture diagram
>
> > ha yes ofc rigid sits under closed.monoidal which replaces the current biclosed, never liked that word anyway
>
> > rename it in the abc too, ClosedMonoidalCategory vs ClosedSymmetricCategory
>
> > move the linear column to the RHS of plain
>
> > not sure what the grey subtitles mean in each box, is that meant to be the abc? just drop the grey subtext and replace it with a picture of a diagram representative of that level
>
> > yes let's go!

- [x] `abc.BiclosedCategory` is `ClosedMonoidalCategory`, `abc.ClosedCategory` is `ClosedSymmetricCategory`, no alias
- [x] `feedback.closed.monoidal`, `feedback.closed.balanced`, `feedback.closed.symmetric`: the free traced categories over the closed levels, no Int at the planar one
- [x] `pivotal`, `ribbon`, `compact`, `frobenius` sit under their `feedback.closed` level: the delay is the identity by default, `Feedback(f)` is `Trace(f)`, an axiom equates the trace with cap, f, cup
- [x] `frobenius`: the delay preserves the spiders
- [x] `Head`, `Tail`, `FollowedBy` are carried down; a functor into a codomain without stream semantics raises on them
- [x] `linear.symmetric` (exchange is the swap by default) and `linear.markov` (contraction and weakening of banged objects are copy and discard, no default bang)
- [WIP] @T-f1b236df-2026-11-26 12:46 architecture pictures drawn by a committed script from each module, `ev` and `λ` labels, linear.symmetric and linear.markov in the diagram
- [ ] docs, changelog, an architecture decision record, references checked against the papers
