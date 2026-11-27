optyx's `QuimbBackend` with a hyperoptimiser keeps the result's axes in wire
order, where it used to permute them once a result had eleven or more input
or output wires ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
