`frobenius.Diagram.unfuse` unfuses the spiders a `Cospan` holds, so
`(Spider(1, 4, x) @ y).unfuse()` and `Diagram.spiders(1, 4, x @ y).unfuse()`
have no point with more than three legs, as upstream ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
