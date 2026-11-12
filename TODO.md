> Review round on #508 (cubic-dev-ai[bot]), head 8b577182:
>
> 1. discopy/optyx/channel.py:1656 (thread 4545470005)
> [P1] The base `Channel` omits `kraus` from `semantic_attrs`, so channels with the same name, dom, cod but different Kraus maps compare and hash equal. Add `"kraus"` to `Channel.semantic_attrs`, the same gap #386 is about.
>
> 2. discopy/optyx/core/zw.py:469 (thread 4545470015)
> [P2] `Create.dagger()` produces a `Select` that carries the same `internal_states`, but `Select` has no `setoid` override. Two `Select`s with different internal states still compare equal and hash the same. Add a matching `setoid` to `Select` that appends `hashable(self.internal_states)`.
>
> 3. discopy/optyx/core/zw.py:474 (thread 4545470019)
> [P2] This doctest passes the states positionally, so they become extra photon counts and `internal_states` stays `None`. The `!=` holds for an unrelated reason and does not test the new `setoid`. Pass them as a keyword: `Create(1, 1, internal_states=([1, 0], [0, 1]))`.
>
> ```suggestion
>         >>> assert Create(1, 1) != Create(1, 1, internal_states=([1, 0], [0, 1]))
> ```
>
> 4. test/optyx/test_channel.py:371 (thread 4545470028)
> [P2] This `zw.Create` case passes only because the positional tuple becomes a third photon, which changes the codomain. `internal_states` is keyword-only, so the pair never tests internal states. Pass it as `internal_states=([1, 0], [0, 1])`.
>
> ```suggestion
>         (zw.Create(1, 1), zw.Create(1, 1, internal_states=([1, 0], [0, 1]))),
> ```
>
> 5. discopy/optyx/utils/misc.py:34 (thread 4545470038)
> [P2] `hashable` raises for zero-dimensional NumPy arrays, so comparing or hashing a 1×1 gate constructed from `np.array(1)` fails. Handle zero-dimensional arrays with `.item()` before iterating arrays.
>
> ```suggestion
>     if isinstance(value, np.ndarray) and value.ndim == 0:
>         return hashable(value.item())
>     if isinstance(value, (list, tuple, np.ndarray)):
>         return tuple(map(hashable, value))
> ```

- [x] discopy/optyx/channel.py:1656 The base `Channel` omits `kraus` from `semantic_attrs`, so channels with the same name, dom, cod but differen…
- [x] discopy/optyx/core/zw.py:469 `Create.dagger()` produces a `Select` that carries the same `internal_states`, but `Select` has no `setoid` o…
- [x] discopy/optyx/core/zw.py:474 This doctest passes the states positionally, so they become extra photon counts and `internal_states` stays `…
- [x] test/optyx/test_channel.py:371 This `zw.Create` case passes only because the positional tuple becomes a third photon, which changes the codo…
- [x] discopy/optyx/utils/misc.py:34 `hashable` raises for zero-dimensional NumPy arrays, so comparing or hashing a 1×1 gate constructed from `np.…
