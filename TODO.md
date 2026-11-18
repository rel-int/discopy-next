> cubic-dev-ai, round on next#629:
>
> > P2 (CHANGELOG.md:416): This changes the return shape of `Diagram.encode`, which breaks existing callers that read its second element as flat `(box, offset)` pairs. Mark it as breaking and say how to migrate, as the `explicit_trace` entry does (e.g. flatten the layers, or call `decode` on the result).
>
> > P3 (docs/notebooks/diagrams.md:500): The sentence just above this line still says 'the full list of layers is redundant', which is no longer true in general. Layer grouping is only redundant when each layer has one box. Consider rewording the preceding sentence to match.

- [ ] CHANGELOG: say encode's new shape breaks callers and how to migrate
- [ ] diagrams notebook: reword both sentences, re-run --check diagrams
