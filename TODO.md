> I got some spare tokens to spend: let’s use them to polish all things discopy-next. Make a deep review of the code and the infra around it, summarising your findings and suggestions in an issue, fixing any bugs that you can fix without requiring a design decisions from my side.
>
> go ahead with the fixes once workflow A lands

The fixes of rel-int/discopy-next#445 in `python` and `drawing`: bugs the review confirmed that need no design decision, one commit and one regression test each.

- [x] **S9-2** Permutation(...) coerces a finset.Function through its cod-length list and ignores its dom, so Permutation.then and Permutation.tensor with a discard …
- [x] **S9-3** Drawing.dagger() raises a bare AssertionError from validate_attributes on every drawing that contains a bubble, frame or slot (including an Equation …
- [x] **S9-5** Drawing.dagger of a drawing with two or more boxes returns a frozen networkx reverse view, so add_nodes, add_edges, add_box_corners and in-place …
- [x] **S9-6** With to_tikz=True, draw(doctest=...) overwrites the .tikz and .tikzstyles baselines instead of comparing against them
- [x] **S9-7** TikZ.output derives the .tikzstyles path by splitting the whole path on dots:
- [x] **S9-8** Drawing an SVG to a BytesIO or to an open file object (text or binary) crashes in add_dark_mode_style after Matplotlib has already written the SVG …
- [x] **S9-10** drawing.draw's docstring, which is also Diagram.draw's, documents two parameters that draw never reads (draw_as_nodes, color) and wrong defaults for …
- [x] **S9-11** Drawing.bubble with non-drawing types (symmetric, rigid, compact, frobenius) whose dom or cod length matches the inside's adds nodes with no …
- [x] **S9-12** Drawing.__eq__ ignores the edges of the plane graph, so drawings with the same nodes and positions but different wiring compare equal;
- [x] **S9-13** additive.Function.braid called on an instance raises TypeError:
- [x] **S9-15** Function.no_type_checking on a subclass writes a permanent type_checking attribute into it:
- [x] **S9-16** The Matplotlib backend draws the label of a TRANSPARENT box or braket white, with no dark-mode tag:
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 full suite and lint gates on the branch
- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-05 19:42 one `CHANGELOG.md` line per user-facing fix
