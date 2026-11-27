`quantum.zx.Hypergraph` keeps spider phases as exact `Fraction`s modulo a
turn (decimal floats included, so `dagger` is an involution on
`Z(1, 1, 0.1)`), `turn` is exact on the quarter-turns so Clifford scalars
round-trip through `to_diagram`, `zx.Spider` writes non-dyadic `Fraction`
phases as `Fraction(p, q)` in `__str__` (keeping `p/q` for dyadic ones) and
exact phases in `__repr__`, and `zx.Scalar.__str__` formats its number
without rounding, with regression cases in `proptest/corpus/zx.py`
([next#718](https://github.com/rel-int/discopy-next/issues/718)).
