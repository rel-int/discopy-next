> Codex on `6306415`:
>
> > **Preserve region colours in the inherited permutation builder**: the inherited `symmetric.Diagram.permutation` still starts its tensor accumulation from the uncoloured `cls.ob()`. Consequently, with `x = Ty('x')` and `R = Box('R', x @ x, x @ x)`, `R.permute(1, 0)` raises `AxiomError` (`None != white`) instead of swapping the outputs. The same builder is used by the inherited cospan/function constructors, so even `Diagram.from_cospan(x, [0], [0, 0])` fails.
>
> > **Keep formal sums in the Peirce diagram category**: for parallel Peirce boxes, the inherited `+` constructs a base-category sum: `alpha` returns `monoidal.Sum`, and `beta` returns `frobenius.Sum`. Thus `M(R + S)` reaches this access with a diagram that has no `colour` property and raises `AttributeError`.

- [x] symmetric.Diagram.permutation starts from the empty type of its first type's colour, so permute and from_cospan work on coloured wires
- [WIP] @session_01MEJxy4BLudMDJBi1HBa5XG-2026-11-19 02:43 alpha.Sum and beta.Sum registered, so a sum stays a Peirce diagram and a model reads it
