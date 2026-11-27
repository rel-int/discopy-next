`reservoir.Readout` compares by value: `eval(repr(r)) == r` (with `array` in
scope), and two identical fits compare equal ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
