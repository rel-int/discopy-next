`optyx.classical.Digit`, `PostselectDigit`, `PostselectBit` and the `Bit`
built on the last reject the empty case with a `ValueError` instead of
reading it as one mode or as the scalar one: a digit of no digits is
`Id()` already ([next#498](https://github.com/rel-int/discopy-next/pull/498)).
