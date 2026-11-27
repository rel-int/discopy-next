The hierarchy of `docs/adr/2026-11-26-traced-levels-with-duals.md`, drawn in
the architecture diagram of the docs by `docs/architecture.py`:
`abc.BiclosedCategory` is `abc.ClosedMonoidalCategory` and
`abc.ClosedCategory` is `abc.ClosedSymmetricCategory`, with no alias.
`feedback.cat` is the free `abc.DelayCategory`, a category with a delay
and no tensor, which `abc.FeedbackCategory` now extends.
`feedback.closed.monoidal`, `feedback.closed.balanced` and
`feedback.closed.symmetric` are new, beside `feedback.closed.markov`: the
free feedback categories over `closed.monoidal`, `feedback.balanced`,
`closed.symmetric` and `closed.markov`, traced once the delay is the
identity, and `pivotal`, `ribbon`, `compact` and `frobenius` sit under them,
with the free delay, which commutes with cups and caps: at delay zero their
feedback is their `Trace`, a box of its own that `Diagram.open_traces`
sends to a cap, the diagram and a cup, as
`abc.PivotalCategory.trace_duals` says, the feedback of a delayed memory
stays a `Feedback` box, and `abc.HypergraphCategory.spider_delay` says the
delay preserves spiders. A functor into a codomain whose delay is the
identity, e.g. `Tensor`, sends a delayed box that is not a key of its
`ar_map` to the image of the box itself. A
category whose trace is not a box, e.g. tensors, circuits or optyx
channels, takes `Diagram.dual_trace` as its `trace_factory`.
`linear.symmetric` takes the swap for the exchange and `linear.markov` the
copy and discard of a banged object for its contraction and weakening,
with no default bang ([next#744](https://github.com/rel-int/discopy-next/pull/744)).
