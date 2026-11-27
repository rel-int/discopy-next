A bubble does not commute with dagger: `cat.Bubble.setoid` counts
`is_dagger`, as `tensor.Bubble`'s has and as `cat.Bubble.__str__` prints,
so that a functor cannot separate equal bubbles. Upstream states the
commuting law, which next `main` now diverges from ([next#507](https://github.com/rel-int/discopy-next/pull/507)).
