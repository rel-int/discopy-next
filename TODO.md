> Review round on #565 (cubic-dev-ai[bot]), head c7fe4292:
>
> 1. test/drawing/typst_rendering.py:166 (thread 4591969437)
> [P2] These tests skip only when `typst` is not installed, but compilation also requires downloading CeTZ; offline environments therefore fail the suite instead of skipping. Gate the integration tests on CeTZ availability or provide a cached/local package for CI.
>
> 2. discopy/drawing/backend.py:1642 (thread 4591969453)
> [P2] Controlled gates lose their control dots and target markers in Typst output because `Typst` inherits the base no-op `draw_node()`. Implement Typst node rendering before drawing controlled boxes.
>
> 3. discopy/drawing/backend.py:1727 (thread 4591969472)
> [P2] The Typst backend drops `style='->'`, so measurement boxes lose their output arrow and render the marker as an ordinary wire. Preserve the arrow style when emitting the Typst primitive.
>
> 4. discopy/drawing/backend.py:1652 (thread 4591969485)
> [P2] Matplotlib color strings such as `tab:blue`, `C0`, and `0.5` are emitted raw and cannot compile as Typst colors. Convert color-like strings to hex before building the Typst expression, while retaining the existing named-color mapping.
>
> ```suggestion
>         return COLORS.get(
>             color, to_hex(color) if is_color_like(color) else color)\
>             if isinstance(color, str) else to_hex(color)
> ```
>
> 5. discopy/drawing/__init__.py:84 (thread 4591969496)
> [P3] The `draw` docstring documents `format` twice, which makes the API reference ambiguous. Merge the Typst description into the existing `format` entry and remove this duplicate entry.
>
> 6. discopy/drawing/typst_ast.py:164 (thread 4591969507)
> [P3] `Import.render` bypasses `Str` when quoting the module path, so paths containing quotes or backslashes produce malformed imports. Render the path through `Str(self.path)` in each branch.

- [x] test/drawing/typst_rendering.py:166 These tests skip only when `typst` is not installed, but compilation also requires downloading CeTZ; offline …
- [x] discopy/drawing/backend.py:1642 Controlled gates lose their control dots and target markers in Typst output because `Typst` inherits the base…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:41 discopy/drawing/backend.py:1727 The Typst backend drops `style='->'`, so measurement boxes lose their output arrow and render the marker as a…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:41 discopy/drawing/backend.py:1652 Matplotlib color strings such as `tab:blue`, `C0`, and `0.5` are emitted raw and cannot compile as Typst colo…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:41 discopy/drawing/__init__.py:84 The `draw` docstring documents `format` twice, which makes the API reference ambiguous.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:41 discopy/drawing/typst_ast.py:164 `Import.render` bypasses `Str` when quoting the module path, so paths containing quotes or backslashes produc…
