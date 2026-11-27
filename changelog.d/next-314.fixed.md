`optyx.core.backends.PercevalBackend` evaluates a term as amplitudes and
applies the Born rule once, to their sum, so a sum of diagrams interferes:
`Create(1) + Create(1) @ Scalar(-1)` is zero where the probabilities of the
two terms added to two. It also carries the two factors that a `path.Matrix`
keeps outside its array and the simulator's circuit therefore cannot: the
global `scalar`, so that the amplitude of `Create(1) @ Scalar(1j)` is `1j`
and not `1`, and the `normalisation` that `dilate` raises to the number of
photons, so that the number operator on two photons is `2` and not the unit
a renormalised state vector reports ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
