`tensor.Bubble.grad` applies the chain rule on any boundary (states, effects,
compound `Dim`s), where it used to raise `ValueError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
