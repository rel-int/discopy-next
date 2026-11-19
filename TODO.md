> Decided small items of #579 (findings from drafting the tool paper): TikZ paints a triangle's label over its sides (no fill=white where it covers); Typst hides a wire label under a later box / fixed box width; docstrings of drawing/__init__.py and drawing/backend.py say "Matplotlib and TikZ" and the autosummary of backend omits Typst; CHANGELOG lists discopy.ehypergraph under Added and renames it under Changed (fix the Added line to the final name); docs/notebooks/categorical-deep-learning.md claim made false by neural.torch.Module(tied, weights); normal_form() fails on a closed term with let.

- [x] Typst draws its labels after its boxes and wires
- [x] CHANGELOG: discopy.egraph under Added, no rename under Changed
- [x] already fixed on main: the TikZ triangle label (#605), the Typst box width, the docstrings and the notebook (8acbe9be)
- [x] normal_form with let: a design question, put in the PR, filed as #579
