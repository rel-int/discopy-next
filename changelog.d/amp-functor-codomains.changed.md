`matrix.Array` holds what `Matrix`, `tensor.Tensor` and
`quantum.channel.Channel` share, the array, its `dtype`, sums, `map`,
`resolvent` and the identity `delay`, and declares no category. `Tensor`
is no longer a `Matrix` but its sibling, a `abc.HypergraphCategory` whose
`merge` and `basis` are its own rather than those of the direct sum, and
`Channel` is no longer a `Tensor` but a dagger `abc.CompactCategory`,
whose `copy` copies in the computational basis, decohering a quantum wire
it is discarded from, and whose `trace` takes a memory `mem=` alone, a
`CQ` type having no last wire. `Tensor.zero` and `Channel.__init__` give
way to `Array.shape` ([next#771](https://github.com/rel-int/discopy-next/pull/771)).
