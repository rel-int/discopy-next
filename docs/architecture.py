"""
Draw docs/api/architecture.html, the table of the free categories that
docs/api/syntax.rst includes, and the picture of each of its cells in
docs/_static/architecture/, each drawn by its own module.

    uv run python docs/architecture.py
"""

import importlib
from pathlib import Path

from discopy import monoidal

DOCS = Path(__file__).parent
PICTURES = DOCS / "_static" / "architecture"

HEAD = [
    ("plain", None, ""), ("linear", "linear", "a bang"),
    ("feedback", "feedback", "a delay and a trace"),
    ("closed", "closed", "exponentials"),
    ("feedback.closed", "feedback.closed", "both"),
    ("duals", None, "trace = cups at delay 0")]
GRID = [
    [["cat"], ["linear.cat"], ["feedback.cat"], [], [], []],
    [["monoidal"], ["linear.monoidal"], ["feedback"],
     ["closed.monoidal"], ["feedback.closed.monoidal"], ["pivotal"]],
    [["spacial"], [], [], ["rigid"], [], []],
    [["braided"], [], [], [], [], []],
    [["balanced"], [], ["feedback.balanced"], [],
     ["feedback.closed.balanced"], ["ribbon"]],
    [["symmetric"], ["linear.symmetric"], ["feedback.symmetric"],
     ["closed.symmetric"], ["feedback.closed.symmetric"], ["compact"]],
    [["markov", "comarkov"], ["linear.markov"],
     ["feedback.markov", "feedback.comarkov"],
     ["closed.markov"], ["feedback.closed.markov"], ["frobenius"]],
]
LABELS = {
    "<<": "ev", ">>": "ev", "**": "ev", "__call__": "ev", "lambda": "λ"}


def picture(name):
    """ The equation that defines a module, drawn by itself. """
    m = importlib.import_module("discopy." + {
        "cat": "monoidal", "linear.cat": "linear.monoidal",
        "feedback.cat": "feedback"}.get(name, name))
    eq = getattr(m, "Equation", monoidal.Equation)
    x, y = m.Ty("x"), m.Ty("y")
    f, g = m.Box("f", x, x), m.Box("g", x, x)
    if name == "cat":
        return eq(m.Id(x) >> f, f, f >> m.Id(x))
    if name == "monoidal":
        return eq(f @ x >> x @ g, x @ g >> f @ x)
    if name == "spacial":
        s = m.Box("s", m.Ty(), m.Ty())
        return eq(s @ x, x @ s)
    if name == "braided":
        return eq(f @ x >> m.Braid(x, x), m.Braid(x, x) >> x @ f)
    if name == "balanced":
        return eq(*(d.to_braided() for d in (
            f >> m.Twist(x), m.Twist(x) >> f)))
    if name == "symmetric":
        return eq(m.Swap(x, y) >> m.Swap(y, x), m.Id(x @ y))
    if name == "markov":
        return eq(m.Copy(x) >> x @ m.Discard(x), m.Id(x))
    if name == "comarkov":
        return eq(x @ m.Merge(x, 0) >> m.Merge(x), m.Id(x))
    if name == "linear.cat":
        return eq(
            m.Digging(x) >> m.Dereliction(x.bang()), m.Id(x.bang()))
    if name == "linear.monoidal":
        return eq(m.Diagram.contraction(x)
                  >> x.bang() @ m.Diagram.weakening(x), m.Id(x.bang()))
    if name == "linear.symmetric":
        contraction = m.Diagram.contraction(x)
        return eq(
            contraction >> m.Diagram.exchange(x, x.bang()), contraction)
    if name == "linear.markov":
        dereliction = m.Dereliction(x)
        return eq(dereliction >> m.Copy(x),
                  m.Copy(x.bang()) >> dereliction @ dereliction)
    if name == "feedback.cat":
        f = m.Box("f", x, y)
        return eq(f >> m.Wait(y), m.Wait(x) >> f.delay())
    if name == "feedback":
        f = m.Box("f", x @ x.delay(), x @ x)
        return eq(
            (f >> x @ g).feedback(), (x @ g.delay() >> f).feedback())
    if name == "feedback.balanced":
        return eq(*(d.to_braided() for d in (
            m.Braid(x, x).trace(), m.Twist(x))))
    if name == "feedback.symmetric":
        return eq(m.Wait(x), m.Swap(x, x.delay()).feedback())
    if name == "feedback.markov":
        fix = (f >> m.Copy(x)).trace()
        return eq(fix, fix >> f)
    if name == "feedback.comarkov":
        cofix = (m.Merge(x) >> f).trace()
        return eq(cofix, f >> cofix)
    if name.startswith("closed."):
        exp = x << y if name == "closed.monoidal" else x ** y
        return eq(m.Eval(exp).curry(), m.Id(exp))
    if name.startswith("feedback.closed."):
        exp = x << x if name.endswith(("monoidal", "balanced")) else x ** x
        ev = m.Eval(exp)
        h = m.Box("h", x @ x, x @ exp @ x)
        return eq((h >> x @ ev).trace(), (x @ ev >> h).trace(2))
    if name == "rigid":
        return eq(x @ m.Cap(x.r, x) >> m.Cup(x, x.r) @ x, m.Id(x))
    if name == "pivotal":
        g = m.Box("f", x, y)
        transpose = m.Cap(x.r, x) @ g.cod.r >> x.r @ g @ g.cod.r\
            >> x.r @ m.Cup(g.cod, g.cod.r)
        return eq(transpose, g.r)
    if name == "ribbon":
        return eq(*(d.to_ribbons() for d in (
            m.Braid(x, x).trace(), m.Twist(x))))
    if name == "compact":
        return eq(m.Swap(x, x).trace(), m.Id(x))
    if name == "frobenius":
        return eq(m.Spider(1, 2, x) @ x >> x @ m.Spider(2, 1, x),
                  m.Spider(2, 2, x))
    raise KeyError(name)


def draw(name):
    """ Draw the picture of a module, with ``ev`` and ``λ`` as labels. """
    drawing = picture(name).to_drawing()
    for node in drawing.nodes:
        if node.kind == "box":
            node.box.drawing_name = LABELS.get(
                node.box.drawing_name, node.box.drawing_name)
    # Drawn to scale, at one unit per 0.4 inch and at least three units high:
    # the images share a height in the table, so they share a scale up to it.
    drawing.draw(path=str(PICTURES / f"{name}.svg"), margins=(.1, .1),
                 wire_labels=False, fontsize=10, aspect="equal",
                 figsize=(.4 * drawing.width, .4 * max(drawing.height, 3)))


def link(name, text=None):
    """ A link to the API page of a module, relative to api/. """
    return f'<a href="../_api/discopy.{name}.html">{text or name}</a>'


def cell(names):
    """ The modules of a cell, each with its picture. """
    items = [f'{link(name)}<br><img src="../_static/architecture/{name}.svg"'
             f' alt="{name}"/>' for name in names]
    if len(names) == 2:
        return "".join(f'<span class="pair">{item}</span>' for item in items)
    return "<br>".join(items)


def table():
    """ The HTML of the table. """
    lines = [
        "<!-- Written by docs/architecture.py and included in",
        "     docs/api/syntax.rst: the links are relative to api/. -->",
        '<table id="architecture-diagram">', "    <thead>", "    <tr>"]
    for title, module, sub in HEAD:
        title = link(module, title) if module else title
        lines.append(f"        <th>{title}<br><small>{sub or '&nbsp;'}"
                     "</small></th>")
    lines += ["    </tr>", "    </thead>", "    <tbody>"]
    for row in GRID:
        lines.append("    <tr>")
        for names in row:
            lines.append(f"        <td>{cell(names) if names else ''}"
                         "</td>")
        lines.append("    </tr>")
    lines += ["    </tbody>", "</table>", ""]
    return "\n".join(lines)


def main():
    """ Draw every picture and write the table. """
    PICTURES.mkdir(parents=True, exist_ok=True)
    for row in GRID:
        for names in row:
            for name in names:
                draw(name)
    (DOCS / "api" / "architecture.html").write_text(table())


if __name__ == "__main__":
    main()
