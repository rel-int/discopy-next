`egraph.EGraph.saturate` runs its rounds on one `Builder`, freezing it
after each round that changes something: associativity and commutativity
of `+` on five leaves saturate 55 times faster, and on six in under a
second ([next#586](https://github.com/rel-int/discopy-next/pull/586)).
