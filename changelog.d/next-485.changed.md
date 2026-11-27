`abc.BraidedCategory`'s `braid_then_inverse`, `inverse_then_braid` and
`inverse_braiding` read the inverse braid from `braid_inverse`, not from
the dagger of the braid, and `dagger_monoidality` is stated by
`abc.DaggerMonoidalCategory`, no longer by `abc.DaggerCategory` ([next#485](https://github.com/rel-int/discopy-next/issues/485)).
