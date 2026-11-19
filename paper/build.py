"""
Build the paper: run its listings, draw its figures, write its LaTeX.

    python -m paper.build          # paper/figures/ and paper/main.tex
    python -m paper.build --pdf    # and main.pdf from Typst, latex.pdf

The ``python`` fences of ``main.typ`` and of the files it includes run in
order in one namespace, as those of the README do; a ``py`` fence is shown
and not run. A ``#drawing("name")`` is the value of ``name`` in that
namespace, drawn to ``figures/name.typ`` by the Typst backend and to
``figures/name.tikz`` by the TikZ one, so that each build of the paper draws
its figures natively. Nothing written here is committed.
"""

import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from paper.transpile import transpile

PAPER = Path(__file__).parent
FENCE = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
INCLUDE = re.compile(r'^#include "([^"]+)"$', re.MULTILINE)
DRAWING = re.compile(r'^#drawing\("(\w+)"', re.MULTILINE)
UNIT = "1cm"


def write(path: Path, text: str) -> None:
    """
    Write a file in one step: the tests build the paper from several
    processes at once, and none may read half a figure.
    """
    with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=path.suffix,
            delete=False) as file:
        file.write(text)
    Path(file.name).replace(path)


def sources(path: Path = PAPER / "main.typ") -> list[Path]:
    """ The source at ``path`` and those it includes, in reading order. """
    text = path.read_text("utf-8")
    return [path] + [
        source for name in INCLUDE.findall(text)
        for source in sources(path.parent / name)]


def run(paths: list[Path]) -> dict:
    """ Run the ``python`` fences of each source, in one namespace. """
    namespace: dict = {}
    for path in paths:
        text = path.read_text("utf-8")
        for match in FENCE.finditer(text):
            padding = "\n" * text.count("\n", 0, match.start(1))
            exec(compile(  # noqa: S102
                padding + match.group(1), str(path), "exec"), namespace)
    return namespace


def draw(paths: list[Path], namespace: dict, figures: Path) -> list[str]:
    """ Draw each ``#drawing`` of the sources with both backends. """
    figures.mkdir(exist_ok=True)
    names = [name for path in paths
             for name in DRAWING.findall(path.read_text("utf-8"))]
    for name in names:
        value = namespace[name]
        drawing = value if hasattr(value, "to_typst") else value.to_drawing()
        document = drawing.to_typst(unit_length=UNIT)
        document.preamble = []
        write(figures / f"{name}.typ", document.render())
        with TemporaryDirectory() as directory:
            tikz = Path(directory) / f"{name}.tikz"
            drawing.draw(to_tikz=True, path=str(tikz))
            write(figures / f"{name}.tikz", tikz.read_text("utf-8"))
    return names


def build() -> list[str]:
    """ Run the listings, draw the figures and write ``main.tex``. """
    paths = sources()
    names = draw(paths, run(paths), PAPER / "figures")
    write(PAPER / "main.tex", transpile(PAPER / "main.typ"))
    return names


def typst() -> bytes:
    """ The PDF of the Typst source, after :func:`build`. """
    from typst import compile as compile_typst
    return compile_typst(str(PAPER / "main.typ"), root=str(PAPER.parent))


def latex() -> Path:
    """ Compile ``main.tex`` to ``latex.pdf``, after :func:`build`. """
    subprocess.run([
        "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
        "-jobname=latex", "main.tex"], cwd=PAPER, check=True,
        capture_output=True)
    return PAPER / "latex.pdf"


if __name__ == "__main__":
    build()
    if "--pdf" in sys.argv[1:]:
        (PAPER / "main.pdf").write_bytes(typst())
        print(PAPER / "main.pdf", latex())
