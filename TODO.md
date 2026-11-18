> 1. **alpha and beta diagrams and their models.** Colours, `Cut`, `alpha.Diagram`, `beta.Diagram`, `Model`; the NPR axioms (both colours cartesian bicategories, linear distributivity, linear adjunctions, de Morgan through the cut) as inequations in the matrix. Cost: ~0 lines changing existing code (more if the colour premise fails), ~450 new with tests, core modules touched: none, unless `frobenius` needs the fix.

- [x] check that frobenius Spider, Swap and Bubble take coloured wires (premise of #628)
- [x] alpha: white, black, Cut, Diagram with ~ & | relative_sum, Model into Tensor[bool]
- [x] beta: Diagram on frobenius, & with white spiders, Model, the NPR axioms of Fig. 9 of BonchiEtAl24 as Inequations enrolled in the matrix
- [WIP] @session_01MEJxy4BLudMDJBi1HBa5XG-2026-11-18 16:38 tests in test/peirce/, doctests: ~~d == d in every model on 1 to 3 elements, each dunder against its definition
- [ ] docs api page entry, CHANGELOG bullet
- [ ] next-checks, proptest -k peirce, CI green with main merged
