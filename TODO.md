> then next step (open a draft PR for this) is to automate the documentation itself: for each level of the syntax we want the equations to be drawn automatically from the axioms of the abstract base class. the best way to do that would be to instantiate the axioms with the category of Drawing where we add an extra attribute for "bracketing" where we draw dotted lines around each operation e.g. associativity (fg)h vs f(gh)

- [x] `Drawing.bracket`: a dotted frame built on `bubble`, with a backend flag for the dotted linestyle (matplotlib and TikZ), no wire label repeated across it, as tight as the drawing inside
- [x] `Drawing.bracketing`: the drawing of a compound term, which `then`, `tensor` and `trace` put in a bracket when it is one of their operands
- [x] `Declaration.generic`: the generic instance of a sequent, its objects and boxes named after its variables and premises, reusing `scope` and `instantiate`
- [x] `discopy.drawing.bracketing.Bracketed[free]`: the terms of a free category with their bracketed drawings, generators delegating to it, `dagger` as a bracket labelled †, `trace` and `feedback` as a loop; `equation(axiom)` draws an axiom in the most general free category that states it
- [x] Sphinx extension in `docs/_ext/` appending an "Axioms" section with each axiom's docstring and its SVG, generated at build time, to each abc class page and to the page of its free category
- [x] Tests: doctest baselines for associativity, dagger contravariance and copy coassociativity, a parametrised test that every stated axiom draws, a CHANGELOG line
- [x] Checks: pflake8, pylint, ruff, ty, pytest, Sphinx build of the touched pages
