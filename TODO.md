> Review round on #565 (cubic-dev-ai[bot]), head 6ada31ff:
>
> 1. test/drawing/typst/braid.typ:8 (thread 4591766727)
> [P1] The Typst backend's `draw_bezier` unpacks the points as `s, e, c1, c2`, but the rest of the backend passes them as start, c1, c2, end. CeTZ's `bezier` reads the second argument as the end point, so each strand gets the wrong geometry. The golden `braid.typ` encodes that wrong output. Fix the unpacking in `Typst.draw_bezier` rather than this file.
>
> 2. test/drawing/typst_rendering.py:166 (thread 4591766738)
> [P2] This test needs the optional `typst` package for the `.svg` branch, but it has no skip guard, so it errors when typst is missing. Add `pytest.importorskip("typst")` before the SVG call, as `test_typst_compilation` does with `pytest.skip`.
>
> ```suggestion
>     pytest.importorskip("typst")
>     svg = f.draw(format="typst", path=tmp_path / "f.svg", show=False)
> ```
>
> 3. test/drawing/typst/box.typ:8 (thread 4591766765)
> [P2] The spacing before `[x]` (and the same gap on the `[y]` and `[f]` lines) doesn't match what `Call.render` emits, which is a single space after the comma. `test_box_to_typst` compares exactly after trimming only trailing whitespace, so this fixture will fail. Regenerate it or use single spaces.
>
> 4. discopy/drawing/typst_ast.py:131 (thread 4591766773)
> [P2] Plain diagram labels are emitted as raw Typst content, so characters such as `]` and `#` can corrupt or execute the generated source. Escape non-math labels before rendering them as content.
>
> 5. pyproject.toml:67 (thread 4591766787)
> [P2] The new Typst tests are not covered by the existing `test` extra. The documented `pip install -e '.[test]'` setup leaves `typst` uninstalled, then the unskipped SVG and IPython rendering tests fail with `ImportError`; add `typst` to the `test` extra.
>
> 6. discopy/drawing/backend.py:1817 (thread 4591766804)
> [P2] Every Typst document imports Fletcher even though the generated body never uses it. Remove the unused import so CeTZ-only rendering does not require an unrelated package.
>
> 7. discopy/drawing/backend.py:1651 (thread 4591766813)
> [P2] RGB sequence colours fail before Typst source generation. Normalize non-string colours to a hex value, while using `COLORS` only for string colour names.
>
> ```suggestion
> return COLORS.get(color, color) if isinstance(color, str) else to_hex(color)
> ```
>
> 8. discopy/drawing/__init__.py:95 (thread 4591766833)
> [P2] Typst skips `frame_dual_rail()`, so dual-rail cups and caps are drawn using bounds that exclude their folds. Frame the drawing after adding box corners, matching `Drawing.draw()`.
>
> ```suggestion
> drawing.add_box_corners()
> drawing.frame_dual_rail()
> ```
>
> 9. discopy/drawing/__init__.py:94 (thread 4591766851)
> [P2] `Drawing.to_typst()` mutates the caller by adding corner nodes directly to its graph. Render a copied drawing, as `Drawing.draw()` does, before adding box corners.
>
> 10. test/drawing/typst_rendering.py:210 (thread 4591766859)
> [P3] This test does not check what its docstring says. Assert the emitted scale value (e.g. `size: 2.0`) and either check the coloured box name or drop it from the docstring.
>
> 11. discopy/drawing/typst_ast.py:7 (thread 4591766874)
> [P3] The example output is stale: it shows fletcher `0.5.9`, but the backend emits `0.5.8`. The autosummary also omits `Bool`, `RawText`, `Ident`, and `Import`. Update the example and add the missing classes.
>
> 12. discopy/drawing/typst_ast.py:63 (thread 4591766891)
> [P3] With `prec=0`, `Float` has no decimal point to anchor the strip, so `Float(10, prec=0)` renders as `1`. Only strip when a decimal point is present.
>
> ```suggestion
>         return s.rstrip('0').rstrip('.') if '.' in s else s
> ```

- [x] test/drawing/typst/braid.typ:8 The Typst backend's `draw_bezier` unpacks the points as `s, e, c1, c2`, but the rest of the backend passes th…
- [x] test/drawing/typst_rendering.py:166 This test needs the optional `typst` package for the `.svg` branch, but it has no skip guard, so it errors wh…
- [x] test/drawing/typst/box.typ:8 The spacing before `[x]` (and the same gap on the `[y]` and `[f]` lines) doesn't match what `Call.render` emi…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/typst_ast.py:131 Plain diagram labels are emitted as raw Typst content, so characters such as `]` and `#` can corrupt or execu…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 pyproject.toml:67 The new Typst tests are not covered by the existing `test` extra.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/backend.py:1817 Every Typst document imports Fletcher even though the generated body never uses it.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/backend.py:1651 RGB sequence colours fail before Typst source generation.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/__init__.py:95 Typst skips `frame_dual_rail()`, so dual-rail cups and caps are drawn using bounds that exclude their folds.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/__init__.py:94 `Drawing.to_typst()` mutates the caller by adding corner nodes directly to its graph.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 test/drawing/typst_rendering.py:210 This test does not check what its docstring says.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:29 discopy/drawing/typst_ast.py:7 The example output is stale: it shows fletcher `0.5.9`, but the backend emits `0.5.8`.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:30 discopy/drawing/typst_ast.py:63 With `prec=0`, `Float` has no decimal point to anchor the strip, so `Float(10, prec=0)` renders as `1`.
