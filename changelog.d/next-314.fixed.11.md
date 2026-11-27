`optyx.core.zw.W` has a `__hash__` of its own: Python used to set it to
`None`, as it does on any class overriding `__eq__`, so `hash(W(2))` raised
and a `Functor` could not look the box up in a dict `ar_map`. `ZBox` stays
unhashable: comparing amplitudes by prefix is not transitive, so no hash
keeps equal boxes hashing equal ([next#314](https://github.com/rel-int/discopy-next/issues/314)).
