An `optyx.core.zw.ZBox` with no inputs keeps every amplitude of a finite
list: its outputs have dimension `max(2, len(amplitudes))` instead of 2, so
`ZBox(0, 1, [1, 2, 3]) >> Select(2)` is 3, as its transpose gives
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
