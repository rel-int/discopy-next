`group.Cyclic`, `Symmetric`, `Product` and `Wreath` compute each action
once and return the same `Action` every time, so a wreath whose top is
itself a wreath costs about as much as its factors (one multiply at nesting
4 went from about 1 s to 0.3 ms). `Action` is a frozen dataclass, so the
action a group shares cannot be changed under its other callers ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
