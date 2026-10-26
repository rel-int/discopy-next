> quantum.graphlike doesn't make much sense as naming, i guess you mean quantum.zx.Hypergraph? refactor it so that the phases are part of the data structure itself, i.e. not boxes but an attribute with composition doing spider fusion plus phase addition, so that the only boxes left are Hadamard

Design, settled with USER in session: phases are generic in `hypergraph.Hypergraph` (an abelian group, summed by the pushout, negated by the dagger), and all of `graphlike` moves into `zx.py` as `zx.Hypergraph`.

- [ ] `hypergraph.Hypergraph.phases`: one per spider, relabelled in `__init__`, summed by `then`, `trace`, `from_glued` and `Rewriter.merge`, negated by `dagger`, forwarded by every constructor
- [ ] equality, hash, `repr`, `to_tree`, `to_graph` and the canonical form read phases; `to_diagram` gives each phased spider a spider box carrying its phase
- [ ] matching: an inner spider matches one of the same phase, a complement keeps what the pattern does not take
- [ ] `zx.Hypergraph`: Z spiders are phased spiders, X spiders green ones behind Hadamards, `H` the only box (scalars kept as boxes)
- [ ] `full_reduce`, `from_pyzx`, `to_pyzx`, `size` as methods of `zx.Hypergraph`, the rules on a `zx.Rewriter`; `graphlike.py` deleted
- [ ] tests, benchmark, docs, CHANGELOG
