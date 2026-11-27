The contextual fraction is piecewise linear in the table and
`quantum.contextuality.Model.gradient` is its gradient, a subgradient at
its kinks: the solution of the dual program, the shadow prices of the
constraints that the solver returns with the value. `ascend` is
gradient ascent on the fraction of any parametrised model, with a step
that doubles after an improvement and halves otherwise since the fraction
is piecewise linear in the table. `Model.normalise` conditions a
postselected model on its success, when that has the same probability in
every context, and `Model.from_diagram` groups several output wires into
one party's outcome. `optyx.contextuality.distributions` contracts the
diagram once with its bit inputs open rather than once per setting, and
`optyx.contextuality.MessagePassingScenario` is the measurement scenario
of a `CMap` for any physical system on its wires: the nodes are the
parties, each with a family of local measurements of its prediction
chosen by a classical wire, `local_measurement` for a qubit and
`dual_rail_measurement` for a pair of modes by default, and `learn` finds
the angles maximising the contextual fraction from the most contextual of
a few random starts. A source node sending half of a fresh Bell pair down
its edge at every tick reaches the Tsirelson bound on qubits and on
dual-rail photons alike ([optyx#87](https://github.com/rel-int/optyx/pull/87)). The notebook
`docs/notebooks/optyx/contextuality.md` shows the calls on a diagram, on
a network and on two photons meeting at a beam splitter, where counting
every event halves the fraction conditioned on a coincidence: the bunched
events are non-contextual and the fraction of the mixture is the
probability of a coincidence times the conditioned fraction, at every
bias of the source.
