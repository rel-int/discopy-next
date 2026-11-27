`feedback.Feedforward` is the dual of `feedback.Feedback`, a bubble whose
memory goes from `mem` to `mem.delay()`: the dagger of a feedback is the
feedforward of the dagger of its argument and back, and `Trace` subclasses
both. `Trace.rotate` is the trace of the rotated argument on the other
side; a feedback or a feedforward that is not a trace still has no
rotation. The dagger involution and contravariance of `feedback` and
`pivotal` diagrams hold, and so does the dagger monoidality of
`symmetric`, `frobenius` and `interaction` ones; the dagger laws of
`markov` and `comarkov` fail on their copies, discards, merges and noises
([next#730](https://github.com/rel-int/discopy-next/pull/730)).
