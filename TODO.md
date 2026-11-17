> cubic P2: `wires()` counts the edges of crossing boxes, which the backend skips, and omits the strands it draws for swaps, braids and permutations. cubic P3: `mean(gaps)` is recomputed for every gap in `whitespace`.

- [x] wires: a crossing's strands from Backend.crossing_wires instead of its edges; whitespace computes the mean once; test
