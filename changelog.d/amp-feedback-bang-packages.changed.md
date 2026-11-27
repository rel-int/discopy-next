`discopy.feedback` and `discopy.linear` are packages, see
`docs/adr/2026-11-25-feedback-and-linear-are-packages.md`. The levels with feedback
are `feedback.balanced`, `feedback.symmetric`, `feedback.markov`,
`feedback.comarkov` and `feedback.closed`; `discopy.balanced`, `symmetric`,
`markov`, `comarkov` and `closed` are no longer feedback or traced, their
wires carry no time step and their `Trace`, `Feedback`, `FollowedBy`,
`Head`, `Tail`, `Diagram.wait` and `closed.Timed` moved with no alias.
`abc.BalancedCategory` is no longer traced, `abc.BalancedTracedCategory`
states its yanking. `pivotal`, `ribbon`, `compact` and `frobenius` keep
their trace of cups and caps. `discopy.bang` is `discopy.linear.monoidal`,
with no alias, over `discopy.linear.cat`, the free `abc.ComonadCategory`,
i.e. bang, dereliction and digging with no tensor; contraction, weakening,
laxator and exchange stay in `linear.monoidal`. `abc.GoISituation` states a
symmetric traced bang and a reflexive object, with no free one. Feedback and linear
do not import each other. `Hypergraph.glue` is the trace of a symmetric
hypergraph with no condition on its category, which `plug` uses to rewrite
([next#737](https://github.com/rel-int/discopy-next/pull/737)).
