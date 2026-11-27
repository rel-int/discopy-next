> #768 open head=f9457f0f discopy.dataflow: head, tail and followed-by leave feedback
> cubic: last review COMMENTED on f9457f0f = head
> reviews: cubic-dev-ai[bot] COMMENTED@f9457f0f
> threads: 6, unresolved 6, on us 6 (ccr route failed: resolution unknown)
>
> 1. test/cat.py:696 [cubic-dev-ai[bot]] thread-root-id=4719190979
> [P1] `test_every_box` still discovers the concrete base `dataflow.Head`, `dataflow.Tail`, and `dataflow.FollowedBy`, so these level-only templates leave them unmapped and make `example()` raise `StopIteration` before the law checks run. Add templates for the base dataflow constructors or explicitly exclude these mixins from discovery.
>
> 2. discopy/messages.py:231 [cubic-dev-ai[bot]] thread-root-id=4719190984
> [P2] This message incorrectly says dataflow diagrams map only into streams, although callable generator mappings explicitly support non-stream codomains. Clarify that non-stream codomains require explicit mappings for the dataflow primitives.
>
> ```suggestion
>     "requires mappings for dataflow primitives in a non-stream codomain."
> ```
>
> 3. docs/api/syntax.rst:16 [cubic-dev-ai[bot]] thread-root-id=4719191004
> [P2] The syntax page now describes `dataflow` as a level beside `feedback`, but the included architecture diagram still omits it. Add the dataflow cells to `docs/architecture.py` and regenerate `architecture.html`, or remove this claim from the paragraph.
>
> 4. discopy/dataflow/__init__.py:104 [cubic-dev-ai[bot]] thread-root-id=4719191009
> [P2] `StreamWire.dagger()` is inherited by `HeadOb` and `TailOb`, but the delegated wire dagger reconstructs them with an `is_dagger` keyword that their constructors reject. Override dagger for these derived wires to dagger `self.arg` and preserve `time_step`, or otherwise make the reconstruction compatible.
>
> 5. docs/api/syntax.rst:16 [cubic-dev-ai[bot]] thread-root-id=4719191014
> [P3] This clause is missing the verb parallel to the preceding clause, so the new sentence reads incorrectly. Change it to ``dataflow`` beside ``feedback`` adds the head, tail and followed-by of streams.
>
> ```suggestion
> ``dataflow`` beside ``feedback`` adds the head, tail and followed-by of streams. A module imports the
> ```
>
> 6. discopy/dataflow/__init__.py:86 [cubic-dev-ai[bot]] thread-root-id=4719191024
> [P3] Validate `is_constant` before storing it; otherwise non-boolean inputs silently change stream constancy semantics despite the documented boolean parameter.
>
> ```suggestion
>         if not isinstance(is_constant, bool):
>             raise TypeError("is_constant must be a bool")
>         self.is_constant = is_constant
> ```

- [ ] test/cat.py:696 base dataflow Head/Tail/FollowedBy need examples or exclusion
- [ ] messages.py NO_DATAFLOW wording
- [ ] syntax.rst: architecture diagram omits dataflow; grammar of the clause
- [ ] dataflow HeadOb/TailOb dagger raises
- [ ] StreamWire validates is_constant
