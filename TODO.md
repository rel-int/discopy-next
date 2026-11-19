> cubic-dev-ai's review of 11-19 on #642 (five threads):
>
> - P1 test/drawing/backend.py:233: the test compiles through the optional `typst` unconditionally.
> - P1 backend.py:1896: a label with several `$...$` segments, such as `$a$ and $b$`, is sent to `latex_math` as one formula.
> - P2 typst_ast.py:354: `2.5` is emitted as `\2.5`.
> - P2 typst_ast.py:387: `\sqrt[3]{x}` consumes `[` as the radicand.
> - P2 typst_ast.py:408: `\text{a \& b}` keeps the backslash.

- [x] the five threads fixed, each with a test
