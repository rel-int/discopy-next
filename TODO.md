> Decided item of #579 (taken as the 48-hour default of the ask in comment 6933004747, asked 11-16, due 11-18, no word from USER): a label between dollars is LaTeX in every backend: the Typst backend translates the LaTeX of a label to Typst math from a table of the symbols it knows (`\mapsto`, `\otimes`, Greek letters, `<=` as LaTeX reads it, etc.), and raises on one it does not know. `Equation(f, g, symbol="$\\mapsto$").draw(format="typst", path=...)` must work.

- [ ] Typst translates the LaTeX of a label between dollars to Typst mathematics, raising on a command it does not know
- [ ] a test: every symbol of the table compiles, and the equation with \mapsto draws
- [ ] CHANGELOG bullet
