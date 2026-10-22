> Move all of the optyx work into discopy-next, keep the license, make sure you keep the attribution of commits, copy all the open issues and pull requests adding giodefelice-agents as assignee. I had planned for him to do it (prompt attached) but I got some spare tokens so I'll do it myself instead.

Copy of rel-int/optyx#88 onto discopy.optyx (#225)

- [x] merge the move branch origin/claude/magical-cannon-g1t6ms
- [x] port to discopy.optyx on next main
- [x] checks
- [ ] giodefelice-agents decides: own repo or notebook (toumix/desire#163)
- [ ] `photon_noise.tables` fails on a cold cache: `json.dumps` cannot write the numpy integers of `np.bincount`, found by `test_photon_noise.py`; fix it or file it along with the decision above
