`Function.no_type_checking` on a subclass of `python.function.Function`
leaves the subclass as it found it, so that the switch on the base class
still reaches it and nesting the two no longer turns its type checking off
for good ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
