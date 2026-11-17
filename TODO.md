> Review round on #578 (chatgpt-codex-connector[bot]), head d14c8429:
>
> 1. discopy/drawing/drawing.py:617 (thread 4593391735)
> **  Handle the empty drawing when computing layout bounds**
>
> For the empty identity `Drawing.id()`, `layout()` returns `()`, but `relayout(())` reaches these `max()` calls with no positions and raises `ValueError: max() iterable argument is empty`. Consequently, the documented `drawing.relayout(drawing.layout())` round trip fails for the monoidal unit. Handle empty positions explicitly or supply appropriate defaults when computing the bounds.
>
> Useful? React with 👍 / 👎.
>
> 2. discopy/drawing/drawing.py:699 (thread 4593391742)
> **  Validate box corners and vertical port attachment**
>
> For a normal `Box('f', 'x', 'x')`, moving only its input port from y=0.75 to y=0.9 is accepted: edge directions remain unchanged and this check considers only x, but the visible input wire now stops above the box. Likewise, moving only `box-corner-01` arbitrarily far away passes because the rectangle checks inspect only corners 00 and 11, although `Backend.draw_box` renders all four corners. Since `add_box_corners` now preserves supplied corners, these invalid geometries persist in the output. Validate that the corners form a nondegenerate box and that its ports remain attached to the appropriate boundary.
>
> Useful? React with 👍 / 👎.
>
> 3. discopy/drawing/drawing.py:694 (thread 4593391750)
> **  Reject new intersections between wires and boxes**
>
> For `(Box('f', 'x', 'x') @ Ty('y')).to_drawing()`, moving both endpoints of the bypassing `y` wire from x=1.5 to x=0.6 passes feasibility: it introduces neither a wire-wire crossing nor a box-box overlap and changes no edge direction. However, the wire now runs through the interior of `f` (whose horizontal bounds are 0.25–0.75), and `Backend.draw_box` paints over its middle. The accepted drawing therefore no longer visibly represents the independent bypass wire. Compare wire-box intersections against the reference as well as box-box overlaps.
>
> Useful? React with 👍 / 👎.
>
> 4. discopy/drawing/drawing.py:612 (thread 4593391754)
> **  Support corner nodes when daggering a relaid-out drawing**
>
> For a drawing containing multiple boxes, such as `(Box('f', 'x', 'x') >> Box('g', 'x', 'x')).to_drawing()`, even `drawing.relayout(drawing.layout()).dagger()` raises `KeyError`. The returned drawing retains the newly added corner nodes, but `Drawing.dagger` only constructs mappings for box, port and boundary nodes, then unconditionally accesses `mapping[n]` for every position. Extend the dagger mapping to reflect and reindex corner nodes so that drawings returned by this API remain usable with the existing drawing operations.
>
> Useful? React with 👍 / 👎.
>
> 5. discopy/drawing/drawing.py:617 (thread 4593391765)
> **  Handle the empty drawing when computingiento layout bounds**
>
> For the empty identity `Drawing.id()`, `layout()` returns `()`, but `relayout(())` reaches these `max()` calls with no positions and raises `ValueError: max() iterable argument is empty`. Consequently, the documented `drawing.relayout(drawing.layout())` round trip fails for the monoidal unit. Handle empty positions explicitly or supply appropriate defaults when computing the bounds.
>
> Useful? React with 👍 / 👎.
>
> 6. discopy/drawing/drawing.py:242 (thread 4593391771)
> **  Convert comparison results before subtracting in sign**
>
> When a layout is supplied as a NumPy floating-point array, for example `drawing.relayout(np.array(drawing.layout()))` for a nonempty drawing, its coordinates remain NumPy scalars. The edge-direction check then calls `sign` with a NumPy scalar, whose comparisons return `numpy.bool_`; subtracting those booleans raises `TypeError` rather than accepting the unchanged layout. This prevents using NumPy vectors directly with the new layout API. Convert the comparison results to Python integers before subtraction, or normalize the supplied coordinates to Python floats.
>
> Useful? React with 👍 / 👎.

- [x] discopy/drawing/drawing.py:617 ** Handle the empty drawing when computing layout bounds** For the empty identity `Drawing.id()`, `layout()` …
- [ ] discopy/drawing/drawing.py:699 ** Validate box corners and vertical port attachment** For a normal `Box('f', 'x', 'x')`, moving only its inp…
- [ ] discopy/drawing/drawing.py:694 ** Reject new intersections between wires and boxes** For `(Box('f', 'x', 'x') @ Ty('y')).to_drawing()`, movi…
- [ ] discopy/drawing/drawing.py:612 ** Support corner nodes when daggering a relaid-out drawing** For a drawing containing multiple boxes, such a…
- [ ] discopy/drawing/drawing.py:617 ** Handle the empty drawing when computingiento layout bounds** For the empty identity `Drawing.id()`, `layou…
- [ ] discopy/drawing/drawing.py:242 ** Convert comparison results before subtracting in sign** When a layout is supplied as a NumPy floating-poin…
