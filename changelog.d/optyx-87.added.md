`quantum.contextuality.Model.from_diagram`, the empirical model of an
optyx diagram with a classical setting and outcome for each party, read
off `optyx.contextuality.distributions` (Apache-2.0) with
`EvalResult.prob_dist(normalise=False)`, with `Model.deterministic`,
`Scenario.bell` taking settings for each party and `Inequality.violation`
zero when the inequality holds, rather than raising on the trivial one
([optyx#87](https://github.com/rel-int/optyx/pull/87)).
