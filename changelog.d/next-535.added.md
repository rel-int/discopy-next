`stream.up_to_steps`, the relation a law of streams is read with: a term
unrolled for `stream.STEPS` steps, its arrow `now` taken and quotiented by
the equation of the underlying category, which `Stream.equation_factory`
compares. Equality on a stream is intensional — `==` compares the thunk
`_later` by identity, so it is identity on any non-constant stream, which
`stream.Ty` now documents — so a law stated with `==` says nothing about
the law. `Ty.strategy`, `Ty.generator_strategy` and `Stream.strategy`
generate the terms those laws quantify over; the property matrix does not
check streams yet, see [next#533](https://github.com/rel-int/discopy-next/issues/533) ([next#535](https://github.com/rel-int/discopy-next/pull/535)).
