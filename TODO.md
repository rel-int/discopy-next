> Review round on #507 (cubic-dev-ai[bot]), head f7ee5926:
>
> 1. discopy/monoidal.py:352 (thread 4545305900)
> [P2] `List.__hash__` still hashes class-valued generators by identity, so documented `List[type]` values remain nondeterministic across interpreter runs. Canonicalize class elements in `self.inside` with `factory_key` before hashing.
>
> ```suggestion
>         return hash((
>             factory_key(type(self)),
>             tuple(factory_key(x) if isinstance(x, type) else x
>                   for x in self.inside)))
> ```
>
> 2. discopy/tensor.py:1096 (thread 4545305920)
> [P2] `tensor.Box` accepts backend dtype descriptors that are not classes, but this unconditionally passes them to `factory_key`. The existing PyTorch `Box[float64]` path therefore raises `AttributeError` whenever equality or hashing invokes `setoid`; use a stable non-class key as well.
>
> ```suggestion
>         dtype = None if self.dtype is None else (
>             factory_key(self.dtype) if isinstance(self.dtype, type)
>             else factory_name(self.dtype))
> ```

- [x] discopy/monoidal.py:352 `List.__hash__` still hashes class-valued generators by identity, so documented `List[type]` values remain no…
- [x] discopy/tensor.py:1096 `tensor.Box` accepts backend dtype descriptors that are not classes, but this unconditionally passes them to …
