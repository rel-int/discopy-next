`Matrix.repeat`, and with it Boolean `Matrix.trace` and `Tensor.repeat`,
takes ceil(log2 n) products instead of n(n+1)/2: milliseconds instead of
seconds at n = 100 ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
