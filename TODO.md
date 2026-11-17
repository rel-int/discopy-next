> Fix rel-int/discopy-next#594: `drawing.nicer`'s hand features ignore dual-rail ribbons. `nicer.wires` reads a drawing's wires as `Backend.visible_edges` plus the strands of crossings (`Backend.crossing_wires`), but `DualRailBraid`, `DualRailTwist`, `DualRailCup` and `DualRailCap` set `draw_as_dual_rail_*` flags, so the ribbons they draw count for nothing in `wire_length`, `bends` and `slope`. Read the curves `Backend.draw_dual_rail_*` draws for those boxes, the way `crossing_wires` gives the strands of a crossing.

- [x] Add `Backend.dual_rail_wires` beside `crossing_wires`, used by `draw_dual_rail_braid`
- [x] `nicer.wires` reads the rails of dual-rail boxes
- [ ] Test in `test/drawing/nicer.py`, failing on main
- [ ] CHANGELOG bullet, next-checks
