`Network.unroll`, the feedforward network running every trace for a
number of rounds from a state of `zeros`, which `Network.to_function` and
`Network.to_lens` take as `rounds`, so that a network with a trace
compiles and its reverse derivative is backpropagation through time: the
reverse derivative preserves the rounds of a trace ([next#518](https://github.com/rel-int/discopy-next/pull/518)).
