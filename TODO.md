> Review round on #605 (cubic-dev-ai[bot]), head faa53c0c:
>
> 1. discopy/drawing/backend.py:1580 (thread 4603020529)
> [P2] TikZ drops the foreground `color` selected for wire and box labels. Because this new branch removes `fill=white` for colored regions, dark regions now render their labels in default black; emit a `text` option from `params["color"]` (using `color_option`) before choosing the fill.
>
> ```suggestion
>         options = "style=none" + (
>             f", text={self.color_option(params['color'])}"
>             if params.get("color") is not None else "") + (
>             ", fill=white" if params.get("adaptive", True) else "")
> ```
>
> 2. CHANGELOG.md:407 (thread 4603020573)
> [P3] The phrase `the width the layout measures a label at fits it` is ungrammatical and obscures the sizing rationale. Say that 8/12 inch fits the label width measured by the layout.
>
> ```suggestion
>   8/12 inch, which fits the label width measured by the layout: every box
>   fits its label, and pictures are 1.7 times larger. A wire of
> ```

- [x] discopy/drawing/backend.py:1580 TikZ drops the foreground `color` selected for wire and box labels.
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-17 17:05 CHANGELOG.md:407 The phrase `the width the layout measures a label at fits it` is ungrammatical and obscures the sizing ration…
