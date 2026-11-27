`optyx.core.path.Matrix` adds formally: `Matrix.sum_factory` is the new
`path.Sum`, `Matrix.zero` is the empty sum and `Sum.eval(n_photons)` adds
the terms' `Amplitudes` on one photon sector, where they add however the
terms create and select photons. It inherited `matrix.Matrix.__add__`,
which added the arrays and dropped `creations`, `selections`,
`normalisation` and `scalar`, so `to_path` of a sum of diagrams raised on
the array shapes — which is how `grad` of a `photonic.Phase` composed with
anything failed — and two matrices of the same shape summed to the wrong
map, their scalars lost. The path matrix of a sum no longer collapses to
one matrix, which only agrees with the sum of the maps on the one-photon
sector, and `optyx.core.diagram.Sum.eval` is that of its path sum rather
than a sum over its terms ([next#693](https://github.com/rel-int/discopy-next/pull/693)).
