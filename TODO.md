> once the whole thing is ready plug it together into a draft paper

USER, 2026-11-16, on the tool paper of #564, whose rulings on its source are: Typst and LaTeX "in the same Ci From the start", "custom code that transpiles this paper adhoc", the snippets compiled and tested there. This pull request is the first slice: the build and the first two sections. The other sections follow in their own pull request.

- [x] paper/: the template, the transpiler from Typst to LaTeX and the build that runs the listings and draws the figures with both backends
- [x] test/paper.py and the paper workflow, both PDFs built in CI
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:28 the introduction and the method section, the other sections as stated gaps
- [ ] CONTRIBUTING and AGENTS, the references, changelog entry, next-checks
