`Circuit.eval(*others, contractor=...)` evaluates every circuit it is given,
so `Sum.eval(contractor=...)` returns the sum of its terms instead of one
term's array summed along its first axis ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
