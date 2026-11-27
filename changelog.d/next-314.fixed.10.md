`optyx.core.zw.ZBox` compares two amplitude lists of different lengths
instead of raising on the broadcast, with
`misc.compare_arrays_of_different_sizes`, which is what equality of two
truncations of the same infinite list is in this module ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
