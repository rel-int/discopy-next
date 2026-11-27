`Box.rebuild` keeps the fields `==` compares, e.g. `tensor.Bubble.func`, a
`tensor.Diff` in `grad`, through `dagger`, `subs` and functors ([next#118](https://github.com/rel-int/discopy-next/issues/118),
[next#169](https://github.com/rel-int/discopy-next/issues/169)).
