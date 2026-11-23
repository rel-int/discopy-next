> Go with the pilot via an Amp session on symmetric, compact and Int

- [WIP] @T-01a2099e-2026-11-23 18:34 a committed corpus format in proptest/corpus/: per cell, arguments written as Python source, eval'd in the module namespace
- [WIP] @T-01a2099e-2026-11-23 18:34 proptest/test_corpus.py replays every entry against its cell, deterministic and offline; an entry that does not build or round-trip through repr fails as a bad entry, not a law
- [ ] pilot corpus for symmetric, written by this Amp session: about 20 terms aimed at shapes the strategies miss
- [ ] pilot corpus for compact: about 20 terms
- [ ] pilot corpus for Int: about 20 terms
- [ ] measure: what the corpus finds that explore did not; file each counterexample as an issue and declare it failing
- [ ] document the corpus and how to grow it in CONTRIBUTING.md, with a changelog entry
