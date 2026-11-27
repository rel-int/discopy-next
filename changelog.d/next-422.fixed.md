`optyx.utils.misc.compare_arrays_of_different_sizes` counts a NaN as a
difference, as `np.allclose` does with its default `equal_nan=False`: it
read `abs(x - y) > tol`, False whenever the difference is a NaN, so a NaN
amplitude compared equal to anything with a matching prefix and an
`optyx.core.zw` axiom stated with the helper could not fail on one. It is
`np.allclose` with an absolute tolerance only instead, so two matching
infinities stay equal, which makes the `zw.has_nan` guard on `ZBox.__eq__`
redundant and both go ([next#422](https://github.com/rel-int/discopy-next/issues/422)).
