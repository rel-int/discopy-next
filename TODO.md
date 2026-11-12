> Review round on #507 (cubic-dev-ai[bot]), head 45c401ff:
>
> 1. discopy/utils.py:508 (thread 4545214074)
> [P2] This key is not injective for user-defined subclasses, so two distinct local `Box`/`Arrow` subclasses can become equal when their qualified names match. Preserve a collision-resistant discriminator for such subclasses, or explicitly restrict this helper to canonical classes whose names are unique; otherwise this changes equality despite the PR’s stated contract.
>
> 2. discopy/utils.py:508 (thread 4545214083)
> [P2] `factory_key` still produces process-randomized hashes because it is an ordinary tuple of strings; every changed `setoid` that hashes this key therefore remains nondeterministic across default interpreter processes. Use a collision-safe key with a deterministic `__hash__`, and ensure nested setoid fields do not reintroduce salted string hashes.

- [x] discopy/utils.py:508 This key is not injective for user-defined subclasses, so two distinct local `Box`/`Arrow` subclasses can bec…
- [x] discopy/utils.py:508 `factory_key` still produces process-randomized hashes because it is an ordinary tuple of strings; every chan…
- [x] CI's own test (3.14) failed on the new test: tensor.Box.setoid holds self.dtype, a class, which is the same mechanism away from the leading component
