> Review round on #565 (cubic-dev-ai[bot]), head b3276cc0:
>
> 1. discopy/drawing/backend.py:1748 (thread 4592064843)
> [P2] Typst draws a straight line when `bend_out` and `bend_in` are both False, but Matplotlib still draws a curve bending through `(source[0], target[1])`. `wire_bezier_points` returns `None` in that case, so the Typst output can differ from the Matplotlib output for the same wire. Confirm the intended behaviour for that case, or make the Typst branch match the Matplotlib control point.
>
> 2. discopy/drawing/__init__.py:92 (thread 4592064856)
> [P2] The Typst dispatch lives only in the module-level `draw`. `Drawing.draw` and `Equation.draw` with `format="typst"` fall through to Matplotlib and fail. Move the dispatch into `backend.draw` or `Drawing.draw` so every entry point behaves the same.
>
> 3. discopy/drawing/__init__.py:94 (thread 4592064867)
> [P2] `format="typst"` breaks the documented in-memory `path` support: `Typst.output` calls `pathlib.Path(path)` for a `StringIO`/`BytesIO` buffer. Handle file-like paths explicitly or reject them with a clear format-specific error.
>
> 4. discopy/drawing/typst_ast.py:60 (thread 4592064877)
> [P2] `Str.render` leaves newline, carriage-return, and tab characters unescaped, so labels containing them can generate invalid or unintended Typst strings. Escape these control characters as `\\n`, `\\r`, and `\\t` alongside quotes and backslashes.
>
> ```suggestion
>         escaped = (
>             self.value.replace("\\", "\\\\")
>             .replace('"', '\\"')
>             .replace("\n", "\\n")
>             .replace("\r", "\\r")
>             .replace("\t", "\\t"))
> ```
>
> 5. discopy/drawing/backend.py:1824 (thread 4592064936)
> [P2] `draw_spiders` ignores `fontsize_spiders`, so spider labels use Typst's default size instead of the drawing API's configured size. Pass the requested size to the `text` call.
>
> ```suggestion
>                     {"fill": Ident(fc),
>                      "size": Ident(
>                          f"{params.get('fontsize_spiders', DEFAULT['fontsize_spiders'])}pt")}
> ```
>
> 6. discopy/drawing/backend.py:1734 (thread 4592064947)
> [P2] Passing a NumPy RGB array as a box color raises `ValueError` here because the array-to-string comparison has ambiguous truth. Use `Backend.is_transparent(facecolor)`.
>
> ```suggestion
>         if facecolor is None or Backend.is_transparent(facecolor):
> ```
>
> 7. discopy/drawing/typst_ast.py:164 (thread 4592064961)
> [P3] `Import.render` silently ignores `alias` when `members` is also set. Either reject the combination, or emit both clauses if Typst allows them. A silent drop is a latent bug for any future caller.

- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:52 discopy/drawing/backend.py:1748 Typst draws a straight line when `bend_out` and `bend_in` are both False, but Matplotlib still draws a curve …
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:52 discopy/drawing/__init__.py:92 The Typst dispatch lives only in the module-level `draw`.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:52 discopy/drawing/__init__.py:94 `format="typst"` breaks the documented in-memory `path` support: `Typst.output` calls `pathlib.Path(path)` fo…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:52 discopy/drawing/typst_ast.py:60 `Str.render` leaves newline, carriage-return, and tab characters unescaped, so labels containing them can gen…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 21:52 discopy/drawing/backend.py:1824 `draw_spiders` ignores `fontsize_spiders`, so spider labels use Typst's default size instead of the drawing A…
- [ ] discopy/drawing/backend.py:1734 Passing a NumPy RGB array as a box color raises `ValueError` here because the array-to-string comparison has …
- [ ] discopy/drawing/typst_ast.py:164 `Import.render` silently ignores `alias` when `members` is also set.
