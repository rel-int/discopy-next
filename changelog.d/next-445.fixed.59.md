`CMap.eval` on a map with no boxes (an identity, cup, cap or loop) uses its
category's dtype, e.g. `tensor.Diagram[complex]`, instead of falling back
to float, so it agrees with `Diagram.eval` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
