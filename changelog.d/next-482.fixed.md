The Matplotlib backend draws a box, a spider, a ket or a bra coloured by an
RGB list or a NumPy array, which matplotlib reads as a colour: it looked
every colour up in `COLORS` and raised `TypeError` on an unhashable one.
`Backend.resolve` and `Backend.is_transparent` look up and compare a colour
only when it is a name, for both backends ([next#482](https://github.com/rel-int/discopy-next/issues/482)).
