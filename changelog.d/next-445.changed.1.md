`kleisli.monad.sample` orders outcomes by the new `kleisli.monad.canonical`,
which sorts the elements of sets and the items of mappings first, so the same
seed draws the same outcome whatever `PYTHONHASHSEED` is, for outcomes built
from tuples, sets and mappings of values whose repr does not depend on
hashing and tells unequal values apart ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
