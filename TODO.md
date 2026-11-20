> toumix approved: file the six bugs as issues on rel-int/discopy-next, then fix them. Start with slices 1–2, the two blocking ones (`head` raising on qubit/qmode/mode, and `double`/`single` dropping `time_step`), which need no design call, each as its own pull request following RULES.md. Leave the design choice between the four options until Giovanni answers his questions. When the issues are filed and the pull requests are open, send their links to thread T-01a1f473-96e6-7657-9227-de1572860147.

- [x] Fix #659: `head` on the wires of `quantum.circuit`, `optyx.channel` and `optyx.core.diagram`, a qubit's head a quantum wire of its dimension
- [ ] Changelog entry
