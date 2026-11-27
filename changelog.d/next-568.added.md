`shape="triangle"` on any box draws it as a triangle, matplotlib and TikZ
alike: its base the width of the side with more wires, its apex on the
other, at its wire when there is one, and its dagger upside down
(`Backend.triangle_corners`). A box's dagger, and any box `rebuild` makes,
keeps the drawing attributes it was given, its `color` and `shape`
([next#568](https://github.com/rel-int/discopy-next/pull/568)).
