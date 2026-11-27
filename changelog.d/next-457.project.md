`stream.Ty.singleton` and `stream.Stream.singleton` pass the callee itself as
`_later` instead of a lambda wrapping it, which takes two of the three
`PLW0108` instances off `discopy/` before the rule graduates out of ruff's
preview, the third being left with [next#443](https://github.com/rel-int/discopy-next/issues/443). Equality still compares
`_later` by identity, so those two now compare equal to a twin built the same
way, where every other non-constant stream does not ([next#457](https://github.com/rel-int/discopy-next/issues/457)).
