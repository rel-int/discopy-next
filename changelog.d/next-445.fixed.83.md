`optyx.core.path.Matrix` compares equal only when its creations, selections,
normalisation and scalar agree too: matrices that evaluate differently used
to compare equal, and matrices with different numbers of creations were
compared by numpy broadcasting or raised `ValueError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
