The TikZ backend escapes in a label what LaTeX reads as markup outside
the mathematics between dollars, an underscore, an ampersand, a percent
and a hash, and keeps `<<` and `>>` from joining into guillemets
(`TikZ.escape`): a box named `feed_forward` or an evaluated lambda term,
drawn as `__call__`, stopped LaTeX ([next#580](https://github.com/rel-int/discopy-next/pull/580)).
