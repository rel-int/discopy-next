> Fix B86's permutation part on rel-int/discopy-next#31, as USER (toumix) ruled live 10-23 (comment 6368666722): a permutation's iterator is turned into a list at init. Today `symmetric.Permutation(x @ y, iter([1, 0]))` is a `Permutation` that is not equal to the `Swap` that the list form `Permutation(x @ y, [1, 0])` gives, because `Permutation.__new__` checks only for `Sequence`.

- [x] Materialise an iterator once in Permutation.__new__ and let each __init__ read the same list
- [ ] Test: iter and generator forms equal the list form, same type and hash
- [ ] CHANGELOG bullet
