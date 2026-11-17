> the ribbon drawing with tikz is broken too, the inside are coloured only for cups and caps
> also wire labels should be shifted down and drawn with a smaller font
> "Dim(4)" should be printed as such but its drawing name should be just "4"
> the min box width should be set properly here so the labels fit

Found while building the tool paper's figures with the TikZ backend. The first point is a box of #579 (ribbons filled differently by each backend).

- [x] TikZ fills the regions of a diagram, the inside of a ribbon included, as Matplotlib does: the loop over region cells moves to the base class
- [x] A TikZ wire label hangs below and to the right of its port, as in Matplotlib, at footnote size, filled white only over an uncoloured region
- [x] `monoidal.Dim.drawing_name` is its numbers, and a wire label uses an object's drawing name when it has one
- [x] Baselines, tests and the changelog
- [x] A TikZ box fits its label: a unit of the picture is LABEL_SIZE / 12 inch, since the layout measures a label at twelve points in inches, and labels are set at fixed sizes, 8 points for a box and 7 for a wire
