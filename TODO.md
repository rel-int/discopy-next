> P2: `Memory[B]` subclasses are not pickleable: pickle looks for `discopy.stream.Memory[B]` as a module attribute, but this factory never registers the generated class. Add a reduce hook or another resolvable registration so diagrams containing `up_to_memory`'s closing boxes can be pickled.

(cubic-dev-ai, review of `4e30866`)

- [x] `Memory[B]` pickled by its subscript, as `NamedGeneric` is; doctest round-trips a box
