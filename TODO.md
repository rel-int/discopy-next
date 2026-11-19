> Codex on `dea5e4b`:
>
> > **Preserve black region colours in composite spider construction**: For a black type `y = beta.Ty(beta.Wire('x', dom=black))`, `beta.Diagram.copy(y @ y)` still raises `AxiomError`: the inherited `frobenius.interleaving` starts from `cls.id()`, which now supplies a white unit, and then tensors it with black spiders. Empty black boundaries also return a white identity instead of preserving their region. Initialize the spider accumulator from the empty type of the supplied boundary rather than relying on this white default.

- [WIP] @session_01MEJxy4BLudMDJBi1HBa5XG-2026-11-19 03:48 frobenius.interleaving starts from the empty type of its boundary, so spiders on a composite or empty black type keep their region
