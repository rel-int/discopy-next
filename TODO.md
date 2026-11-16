> Review round on #577 (cubic-dev-ai[bot]), head 103fe4bb:
>
> 1. paper/sections/method.typ:72 (thread 4593219736)
> [P1] `#drawing("snake")` looks up `snake` in the build namespace, but its definition is in a ` ```py` fence and `paper.build.run` executes only ` ```python` fences. The paper build therefore raises `KeyError` before generating the figures; make this snippet executable or otherwise add `snake` to the run namespace.
>
> 2. CONTRIBUTING.md:348 (thread 4593219741)
> [P2] The documentation says the LaTeX test is skipped when the required packages are missing, but the test checks only for `latexmk`; an installed `latexmk` with a missing package makes the test fail. Clarify that only the absence of `latexmk` is skipped, or add an explicit prerequisite check.
>
> ```suggestion
> the test is skipped only when `latexmk` is absent; missing LaTeX packages still make it fail. A `python` fence of the paper runs, a `py` fence is shown and not run, and
> ```
>
> 3. .github/workflows/paper.yml:48 (thread 4593219763)
> [P2] `groups: all` installs unrelated heavyweight extras such as Torch, Qiskit, JAX, and Sphinx, making this paper job unnecessarily slow and vulnerable to its 20-minute timeout. Request only `dev typst` for the listings, tests, and Typst compiler.
>
> ```suggestion
>         groups: dev typst
> ```
>
> 4. .github/workflows/paper.yml:11 (thread 4593219771)
> [P2] This pull-request filter omits library modules that the paper listings execute, so those changes can leave the paper build untested and its generated PDF stale. Include `discopy/**` (or every module used by the listings) in the filter.
>
> ```suggestion
>     - discopy/**
> ```
>
> 5. paper/transpile.py:154 (thread 4593219784)
> [P2] `call` accepts any `#link` prefix and slices it as `#link("url")`, so unsupported forms such as `#link[bad]` are silently emitted as a different URL. Validate the expected `#link("...")` syntax before extracting the URL and raise `Unsupported` otherwise.
>
> 6. paper/transpile.py:44 (thread 4593219791)
> [P2] `text_escape` leaves LaTeX-special characters such as `_`, `$`, `~`, and `\` unescaped, so plain Typst text can produce invalid or semantically different LaTeX. Add the missing escapes to keep the transpiler's no-drift guarantee.
>
> 7. paper/preamble.tex:4 (thread 4593219799)
> [P3] The LaTeX build uses 10pt body text, but the Typst template uses 10.5pt, so the two paper versions drift in typography and pagination. Set the LaTeX normal font size to 10.5pt to match the Typst layout.

- [x] paper/sections/method.typ:72 `#drawing("snake")` looks up `snake` in the build namespace, but its definition is in a ` ```py` fence and `p…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 CONTRIBUTING.md:348 The documentation says the LaTeX test is skipped when the required packages are missing, but the test checks …
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 .github/workflows/paper.yml:48 `groups: all` installs unrelated heavyweight extras such as Torch, Qiskit, JAX, and Sphinx, making this paper…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 .github/workflows/paper.yml:11 This pull-request filter omits library modules that the paper listings execute, so those changes can leave th…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 paper/transpile.py:154 `call` accepts any `#link` prefix and slices it as `#link("url")`, so unsupported forms such as `#link[bad]` …
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 paper/transpile.py:44 `text_escape` leaves LaTeX-special characters such as `_`, `$`, `~`, and `\` unescaped, so plain Typst text c…
- [WIP] @session_01MEJxy4FrWd3ymhMeQGF7x8-2026-11-16 23:40 paper/preamble.tex:4 The LaTeX build uses 10pt body text, but the Typst template uses 10.5pt, so the two paper versions drift in t…
