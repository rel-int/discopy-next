`Circuit.measure` and `init_and_discard` prepare a digit of any dimension in
`Digits(0, dim=...)`, so a classical circuit on `Digit(4)` measures instead
of raising `AxiomError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
