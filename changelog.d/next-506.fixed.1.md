Every hash in the library is the same in every interpreter:
`utils.factory_key`, a class's module and qualified name, leads a `setoid`
or a `__hash__` where `type(self)` did, the hash of a class object being
its address. No equality changes, a subscript being part of a qualified
name; `tensor.Bubble` is left, holding the function it maps with
([next#506](https://github.com/rel-int/discopy-next/issues/506), [next#507](https://github.com/rel-int/discopy-next/pull/507)).
