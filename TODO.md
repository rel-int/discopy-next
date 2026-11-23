> in fact my goal isn't to implement normal forms everywhere but to define correctness of the drawing algorithm, i.e. different drawings of the same diagram are isotopic and the gif is that proof, ping the discopy-lean space to start proving the result for monoidal while you start working on braided and symmetric gifs

- [x] Settle whether the straight-line tween of one interchanger is a planar isotopy: it is not, `Isotopy.interchanger_overlaps` (sent to the drawing thread)
- [x] Prove what does hold: two boxes separated along one side in both drawings stay apart, `Isotopy.stays_apart`
- [ ] Rule on the fix, in the drawing thread's PR (#706): a layout keeping the boxes of each interchanger separated along one side, or a tween through an intermediate drawing
- [x] Horizontal phases of the fixed tween keep boxes apart: `Isotopy.slides_apart`
- [ ] Wires: no two wire segments cross along a tween (failed on 6f002af1 and df2eb2ca, no counterexample on 18199979 in a sweep of 800 random diagrams; still to prove), and the order of wires around each box and along the boundary is kept
> let's start with a proof of the previous algorithm ie without any fancy optimisation just a monoidal category into plane graphs

- [x] The layered drawing, one box per layer and wires on a unit grid, sends every diagram to a plane graph and composition to stacking: `Diagram.drawing`
- [ ] Interchange on the layered drawing: the two drawings of an interchanger are isotopic, by a tween through a strip where both boxes sit side by side
- [ ] Model `to_drawing` of a whole diagram, and compose the steps of `Equation.path` into an isotopy between any two drawings of one diagram
- [x] Changelog entry
