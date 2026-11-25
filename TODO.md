> Codex review of 55cf7c85 (https://github.com/rel-int/discopy-next/pull/728#pullrequestreview-6061250547):
>
> > Apply the category direction to rich-display and Typst exports: `RichDisplay.to_svg()` and `to_png()` and `drawing.to_typst()` call `diagram.to_drawing()` directly and never propagate `draw_left_to_right`.
>
> > Include nested boxes when computing default asymmetry: a daggered, transposed or conjugated box inside a direction-changing bubble draws like its dagger.
>
> > Preserve arrowheads when transposing measurement symbols: `Transposed.draw_wire` ignores `style='->'`.

- [x] Category direction in to_svg, to_png and to_typst
- [x] Asymmetry counts nested boxes
- [ ] Transposed keeps arrow-style wires
