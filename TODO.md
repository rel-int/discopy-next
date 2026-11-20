> Yes, remove the grammar alias, then start the SPINDLE port

- [WIP] @T-77611241-2026-11-20 20:13 lambek.abstract.spindle: the SPINDLE tagger, its ghosts and Parser.from_pretrained, a sentence read as a ProofNet as Tommaso's parse_with_ghosts does at word_samples=0
- [WIP] @T-77611241-2026-11-20 20:13 check it net for net against Tommaso's spindle/parser.py (wiki#200, c81a8dc) on a random small model
- [ ] check it on rel-int/spindle-ccg-deberta-v3-large on WSJ 23 sentences, in an orb with more than 1 GB
- [ ] temperature resampling of the words (word_samples > 0), which Tommaso's evaluation sweeps
