> P2: These modulo laws now crash for the supported `Stream[python.Function]` instantiation because `python.Function` has no `box_factory`. Make the memory-permutation comparison use an abstraction available to every supported underlying category, or restrict this override to categories that provide the required state/effect construction.

> P2: `up_to_memory` uses the fixed `MEMORY` box name without rejecting collisions with user boxes. A stream containing a legitimate box named `memory` can therefore be mistaken for the sentinel closure during hypergraph quotienting, producing false equalities; validate the name collision or generate a fresh sentinel name.

(cubic-dev-ai, review of `effc773`)

- [x] `up_to_memory` over a category with no `box_factory` is `up_to_steps`, tested on `Stream[python.Function]`
- [x] `MEMORY` a name no user box carries, tested against a box named "memory"
