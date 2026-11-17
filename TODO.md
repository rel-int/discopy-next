> Codex P2: `wires` drops the visible wires of cups, caps, swaps and spiders, which `Backend.visible_edges` keeps. Codex P2: `misalignment` measures the box node, which `draw` recentres, so it penalises what is never seen.

- [ ] wires are Backend.visible_edges; misalignment is the offset between the centres of a box's inputs and outputs; tests
