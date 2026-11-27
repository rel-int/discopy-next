A feed-forward component converted from perceval applies its default
circuit **once**, when the measured state is in none of its configurations,
where each configuration that did not match used to apply it: with a
`PERM([1, 0])` default and two configurations, one of which matched, the
swap was applied anyway. An even number of non-matching entries composes a
swap default into the identity, which is why a single configuration and
three of them — none and two non-matching entries — both read right
([next#314](https://github.com/rel-int/discopy-next/issues/314)).
