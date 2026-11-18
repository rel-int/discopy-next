> P1: `up_to_memory` assumes the input memory is the suffix of `now.dom`, but `Stream.unroll` places it before the newest input. It closes an ordinary input instead of the memory, so nonempty-memory comparisons fail at composition; reorder the unrolled boundary or compose the state boxes at their actual memory positions before closing it.

> P3: This dynamically generated box class breaks the package's repr/round-trip contract: its repr names `discopy.stream.Memory`, but that class is not exported or resolvable from the module. Expose a stable per-factory class name or provide a resolvable reconstruction path before returning these boxes.

(cubic-dev-ai, review of `b36217f`)

- [x] P1 checked: `Stream.__init__` enforces `now.dom == dom.now @ mem.now`, so the memory is the suffix; tested with input and memory
- [x] P3 `Memory[B]` a module-level class whose subscript is the box class, `eval(repr(box)) == box`
