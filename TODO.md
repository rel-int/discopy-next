> once the whole thing is ready plug it together into a draft paper

USER, 2026-11-16, on the tool paper of #564. Its LaTeX figures are drawn by the TikZ backend, which writes labels unescaped (#579, second box): a bug of a next repo, fixed here without asking.

- [x] TikZ escapes in a label what LaTeX reads as markup outside mathematics, test, changelog entry, next-checks
