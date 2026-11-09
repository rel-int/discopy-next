> Review round on #1 (cubic-dev-ai[bot]), head d0769779:
>
> 1. discopy/interaction.py:236 (thread 4491728069)
> [P3] `Ty[int]` is a supported interaction type, but this property assumes every natural half has `is_atomic`; `Ty[int](1, 2).is_atomic` raises `AttributeError`. Use a safe fallback for natural values without that category property.
>
> ```suggestion
> +        return all(getattr(half, "is_atomic", False)
> +                   for half in (self.positive, self.negative))
> ```
>
> 2. discopy/abc.py:1470 (thread 4492193129)
> [P1] These braided-category declarations assume every braid has a dagger inverse, but `BraidedCategory` provides neither operation. Define an inverse-braid API or restrict these laws and rules to a dagger-compatible braided structure.
>
> 3. discopy/abc.py:251 (thread 4492193135)
> [P2] `DaggerCategory` does not provide a tensor, so this inherited axiom cannot run for an ordinary dagger category. Move dagger monoidality to a dagger-monoidal abstraction or make it inapplicable here.
>
> 4. discopy/feedback.py:475 (thread 4492193143)
> [P3] Setting `box_factory = None` on feedback `Ty` overrides the wire generator that `monoidal.Ty` exposes, and the docstring is wrong: `Wire` objects are named. It also makes `test_canonical_instances` skip this type. Remove the override and its docstring, or document why `Ty` has no canonical instance.

- [x] discopy/interaction.py:236 is_atomic on a Ty[int] half
- [ ] discopy/abc.py:1470 braided declarations use a dagger inverse BraidedCategory does not provide
- [ ] discopy/abc.py:251 dagger_monoidality on DaggerCategory, which has no tensor
- [ ] discopy/feedback.py:475 box_factory = None on feedback.Ty
