> toumix approved: file the six bugs as issues on rel-int/discopy-next, then fix them. Start with slices 1–2, the two blocking ones (`head` raising on qubit/qmode/mode, and `double`/`single` dropping `time_step`), which need no design call, each as its own pull request following RULES.md. Leave the design choice between the four options until Giovanni answers his questions. When the issues are filed and the pull requests are open, send their links to thread T-01a1f473-96e6-7657-9227-de1572860147.

- [WIP] @T-43eb8dbf-2026-11-20 15:39 Fix #660: optyx `Ob.single` and `Ob.double` keep the wire's `time_step`
- [ ] Changelog entry
