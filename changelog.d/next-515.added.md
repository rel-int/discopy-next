`Network.to_lens`, the reverse derivative of a feedforward network as a
`neural.network.Lens`, i.e. a lens of functions on tensors whose `get` is
`to_function` and whose `put` is backpropagation: every box gives its
`Box.tape`, composition is the chain rule of `lens.Symmetric` with the
tapes as residual, a copy sums the gradients of its branches and a
discard gives zero ([next#515](https://github.com/rel-int/discopy-next/pull/515)).
