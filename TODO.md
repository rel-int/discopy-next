> #768 open head=2b2aab3d discopy.dataflow: head, tail and followed-by leave feedback
> cubic: last review APPROVED on 2b2aab3d = head
> reviews: cubic-dev-ai[bot] APPROVED@2b2aab3d, chatgpt-codex-connector[bot] COMMENTED@2b2aab3d
> threads: 9, unresolved 9, on us 9 (ccr route failed: resolution unknown)
>
> 1. test/cat.py:698 [cubic-dev-ai[bot]] thread-root-id=4719190979 [answered, resolve it] [addressed e3d7d7f]
> [P1] `test_every_box` still discovers the concrete base `dataflow.Head`, `dataflow.Tail`, and `dataflow.FollowedBy`, so these level-only templates leave them unmapped and make `example()` raise `StopIteration` before the law checks run. Add templates for the base dataflow constructors or explicitly exclude these mixins from discovery.
>   ↳ toumix-agents: Real: since the mixins extend the feedback classes, the census found them. Fixed in e3d7d7f: `dataflow.Head`, `dataflow.Tail` and `dataflow.FollowedBy` go in `ABSTRACT`, now also matched by factory n…
>
> 2. discopy/messages.py:231 [cubic-dev-ai[bot]] thread-root-id=4719190984 [outdated] [answered, resolve it] [addressed e3d7d7f]
> [P2] This message incorrectly says dataflow diagrams map only into streams, although callable generator mappings explicitly support non-stream codomains. Clarify that non-stream codomains require explicit mappings for the dataflow primitives.
>
> ```suggestion
>     "requires mappings for dataflow primitives in a non-stream codomain."
> ```
>   ↳ toumix-agents: Taken in e3d7d7f, reworded: a dataflow functor into such a codomain needs the primitives in its mapping, or a codomain of streams.
>
> 3. docs/api/syntax.rst:16 [cubic-dev-ai[bot]] thread-root-id=4719191004 [outdated] [answered, resolve it] [addressed e3d7d7f]
> [P2] The syntax page now describes `dataflow` as a level beside `feedback`, but the included architecture diagram still omits it. Add the dataflow cells to `docs/architecture.py` and regenerate `architecture.html`, or remove this claim from the paragraph.
>   ↳ toumix-agents: Fixed in e3d7d7f: `docs/architecture.py` gets a dataflow column, `dataflow.symmetric` drawn as `fby` and `dataflow.markov` as a constant stream (fby then copy, fed back); `architecture.html` regenera…
>
> 4. discopy/dataflow/__init__.py:106 [cubic-dev-ai[bot]] thread-root-id=4719191009 [answered, resolve it] [addressed e3d7d7f]
> [P2] `StreamWire.dagger()` is inherited by `HeadOb` and `TailOb`, but the delegated wire dagger reconstructs them with an `is_dagger` keyword that their constructors reject. Override dagger for these derived wires to dagger `self.arg` and preserve `time_step`, or otherwise make the reconstruction compatible.
>   ↳ toumix-agents: Real, reproduced: `x.head.dagger()` raised a TypeError. Fixed in e3d7d7f: the head and tail of a wire dagger to the head and tail of its dagger with the same time step, with a doctest and `test_HeadO…
>
> 5. docs/api/syntax.rst:16 [cubic-dev-ai[bot]] thread-root-id=4719191014 [outdated] [answered, resolve it] [addressed e3d7d7f]
> [P3] This clause is missing the verb parallel to the preceding clause, so the new sentence reads incorrectly. Change it to ``dataflow`` beside ``feedback`` adds the head, tail and followed-by of streams.
>
> ```suggestion
> ``dataflow`` beside ``feedback`` adds the head, tail and followed-by of streams. A module imports the
> ```
>   ↳ toumix-agents: Taken in e3d7d7f.
>
> 6. discopy/dataflow/__init__.py:88 [cubic-dev-ai[bot]] thread-root-id=4719191024 [answered, resolve it] [addressed e3d7d7f]
> [P3] Validate `is_constant` before storing it; otherwise non-boolean inputs silently change stream constancy semantics despite the documented boolean parameter.
>
> ```suggestion
>         if not isinstance(is_constant, bool):
>             raise TypeError("is_constant must be a bool")
>         self.is_constant = is_constant
> ```
>   ↳ toumix-agents: Fixed in e3d7d7f with `assert_isinstance(is_constant, bool)`, as `feedback.Wire` did before the move; tested.
>
> 7. test/cat.py:699 [chatgpt-codex-connector[bot]] thread-root-id=4719228714 [answered, resolve it]
> **  Handle the base dataflow mixins in the box conformance tests**
>
> `box_classes()` also discovers `dataflow.Head`, `dataflow.Tail`, and `dataflow.FollowedBy`: they subclass `cat.Box`, are defined in an imported package, and are neither abstract nor excluded by `ABSTRACT`. These new templates cover only the symmetric and Markov subclasses. For the base mixins, `example()` instead selects the inherited `feedback.Box` template and fails its `issubclass(base, owner)` assertion, aborting every `test_every_box` parametrization before conformance can be checked. Add suitable handling for these base mixins, or explicitly exclude them if they are not intended to be instantiated.
>
> Useful? React with 👍 / 👎.
>   ↳ toumix-agents: Same finding as cubic's on line 696, fixed in e3d7d7f: the three mixins are excluded by factory name in `ABSTRACT`.
>
> 8. discopy/dataflow/__init__.py:406 [chatgpt-codex-connector[bot]] thread-root-id=4719325756
> **  Honor explicit head and tail mappings before the bubble fallback**
>
> When the codomain is a diagram category without dataflow operations, such as `discopy.symmetric.Diagram`, supplying an explicit mapping for `f.head` or `f.tail` passes this check but does not use that mapping. Both primitives have `args=(f,)` and the default `method="bubble"`, so `monoidal.Functor.__call__` handles them as bubbles before consulting `ar_map`. For example, a functor mapping `x.head` to `y` and `f.head` to a box `g: y → y` still tries to evaluate the unmapped `f` and raises instead of returning `g`. Dispatch these explicitly mapped primitives through the generator mapping rather than the generic bubble fallback, as the new non-dataflow-codomain contract promises.
>
> Useful? React with 👍 / 👎.
>
> 9. discopy/dataflow/symmetric.py:192 [chatgpt-codex-connector[bot]] thread-root-id=4719325768
> **  Bind term factories to the new dataflow categories**
>
> For diagrams containing only wiring, the inherited `to_term()` operation evaluates back into the feedback category rather than the dataflow category. Neither new level initializes its own `term_factory`, so it inherits the feedback level's terms; when a term has no constants, `cat.TermBase.category` falls back to that inherited term class's `functor.cod`. Consequently, even `d = Diagram.id(Ty('x'))` does not satisfy the documented `d.to_term().eval() == d` round trip, and constant-free Markov wiring is reconstructed using feedback factories. Initialize the terms for both dataflow levels with `cat.kappa_calculus(Diagram)`, as their feedback parents do, so evaluation retains the originating category.
>
> Useful? React with 👍 / 👎.

- [x] dataflow/__init__.py:406 mapped head/tail go to their images, not bubbles
- [ ] dataflow/symmetric.py:192 term factories bound to both dataflow levels
