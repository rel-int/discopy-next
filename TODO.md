> Add a Feedforward class to discopy.feedback as the dual of Feedback, for https://github.com/rel-int/discopy-next/issues/710. Do the work in this thread, and do not start another one.
>
> Toumix settled the design.
>
> Feedforward is the box that a feedback's dagger draws. Feedback.dagger returns Feedforward(arg.dagger(), ...), and Feedforward.dagger returns the Feedback, so dagger_involution holds by construction. The memory of a feedforward goes from mem to mem.delay() in the opposite direction from a feedback.
>
> Trace is a subclass of both, Trace(Feedback, Feedforward). Both bases should inherit Box and Bubble in the same order so the method resolution order exists, with Feedback first so Trace's existing overrides still win. Trace.dagger stays overridden and returns a Trace. Once Feedback.dagger returns a Feedforward, the inherited method would turn a trace's dagger into one.
>
> Feedback.rotate and Feedforward.rotate keep raising. Only Trace defines rotate, and it flips which side the loop is on:
>
> def rotate(self, left=False):
>     return self.arg.rotate(left=left).trace(
>         len(self.mem), left=not self.left)
>
> The two left arguments are distinct. rotate passes its own argument down to the argument, choosing the direction of the half-turn. The trace side is the negation of the attribute self.left, so a left trace rotates to a right trace and a right trace to a left one. dagger keeps the side, self.arg.dagger().trace(..., left=self.left). rotate is that call with the side flipped.
>
> A half-turn of a delayed feedback has the same shape as its dagger, a feedforward of the rotated argument. Do not implement that. Toumix chose that the two superclasses raise, and that the side flip lives only on Trace, where the delay is the identity.
>
> Feedforward does not make sense for streams in general, only for bidirectional ones. Stream in discopy/stream.py is a FeedbackCategory with no dagger. Do not give Stream an interpretation of Feedforward, and do not add a bidirectional stream class.
>
> feedback.structural_boxes re-exports Feedback and Trace. pivotal.Diagram declares rotate_contravariance and adjunction failing with FEEDBACK_HAS_NO_DAGGER because Feedback.rotate raises. A Trace drawn there should rotate. A Feedback that is not a Trace still should not, so those laws stay failing on a delayed feedback.
>
> The issue still asks what a feedforward means in stream.Stream and whether rotate gets the same treatment. Those two questions are settled as above, so record that on the issue.

- [WIP] @T-51236a2e-2026-11-25 15:06 Feedforward in discopy.feedback: Feedback.dagger and Feedforward.dagger are each other's
- [WIP] @T-51236a2e-2026-11-25 15:06 Trace(Feedback, Feedforward), with Trace.rotate flipping the side of the loop
- [WIP] @T-51236a2e-2026-11-25 15:06 structural_boxes generates each level's Feedforward
- [ ] drop the failing dagger declarations that now hold; keep pivotal's rotate ones
- [ ] record the stream and rotate rulings on #710
- [ ] CHANGELOG bullet and next-checks
