> Fix rel-int/discopy-next#249 by taking the default of our ask there (comment 6484303463, asked 10-29, default: delete), now past its 48-hour due date with no word from USER: delete the notebook `docs/notebooks/optyx/test_light_cones.md`, which imports modules removed from optyx and is the prototype of the light-cone bound now in `discopy/optyx/utils/misc.py` (used by `core.diagram.Diagram.to_tensor`), and remove its entry from `KNOWN_BROKEN` in `docs/export_notebooks.py`.

- [x] Check nothing links to or includes the notebook
- [ ] Cover with a unit test of misc.py whatever the notebook checks that no test does
- [ ] Delete the notebook and its KNOWN_BROKEN entry, with a CHANGELOG bullet
- [ ] Run next-checks and an optyx notebook
