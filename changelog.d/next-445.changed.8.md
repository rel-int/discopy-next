A bubble's `data` is part of its identity, as a box's is: bubbles that
differ only in `data` are unequal, and `str` prints the `data` so that
`eval(str(b)) == b` still holds. As for a box, a bubble holding unhashable
`data` is unhashable, and comparing bubbles whose `data` are arrays of more
than one entry raises `ValueError`, except at the tensor level, where a
bubble compares array and list `data` by their entries ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
