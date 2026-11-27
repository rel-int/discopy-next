"""
Draw docs/api/architecture.html, the table of the free categories that
docs/api/syntax.rst includes, and the picture of each of its cells in
docs/_static/architecture/, each drawn by its own module.

    uv run python docs/architecture.py
"""

import importlib
from pathlib import Path

DOCS = Path(__file__).parent
PICTURES = DOCS / "_static" / "architecture"

HEAD = [
    ("plain", None, ""), ("linear", "linear", "a bang"),
    ("feedback", "feedback", "a delay and a trace"),
    ("dataflow", "dataflow", "head, tail, fby"),
    ("closed", "closed", "exponentials"),
    ("feedback.closed", "feedback.closed", "both"),
    ("duals", None, "trace = cups at delay 0")]
GRID = [
    [["cat"], ["linear.cat"], ["feedback.cat"], [], [], [], []],
    [["monoidal"], ["linear.monoidal"], ["feedback"], [],
     ["closed.monoidal"], ["feedback.closed.monoidal"], ["pivotal"]],
    [["spacial"], [], [], [], ["rigid"], [], []],
    [["braided"], [], [], [], [], [], []],
    [["balanced"], [], ["feedback.balanced"], [], [],
     ["feedback.closed.balanced"], ["ribbon"]],
    [["symmetric"], ["linear.symmetric"], ["feedback.symmetric"],
     ["dataflow.symmetric"], ["closed.symmetric"],
     ["feedback.closed.symmetric"], ["compact"]],
    [["markov", "comarkov"], ["linear.markov"],
     ["feedback.markov", "feedback.comarkov"], ["dataflow.markov"],
     ["closed.markov"], ["feedback.closed.markov"], ["frobenius"]],
]
LABELS = {
    "<<": "ev", ">>": "ev", "**": "ev", "__call__": "ev", "lambda": "λ",
    "FollowedBy(x)": "fby"}


def picture(name):
    """ The characteristic diagram of a module, drawn by itself. """
    m = importlib.import_module("discopy." + {
        "cat": "monoidal", "linear.cat": "linear.monoidal",
        "feedback.cat": "feedback"}.get(name, name))
    x = m.Ty("x")
    f = m.Box("f", x, x)
    if name == "cat":
        return f >> m.Box("g", x, x)
    if name == "monoidal":
        return f @ m.Box("g", x, x)
    if name == "spacial":
        s = m.Box("s", m.Ty(), m.Ty())
        return m.Equation(s @ x, x @ s)
    if name in ("braided", "linear.symmetric"):
        return m.Diagram.braid(x, x) if name == "braided"\
            else m.Diagram.exchange(x, x)
    if name == "balanced":
        return m.Twist(x).to_braided()
    if name == "symmetric":
        return m.Swap(x, x)
    if name == "markov":
        return m.Copy(x)
    if name == "comarkov":
        return m.Merge(x)
    if name == "linear.cat":
        return m.Dereliction(x)
    if name in ("linear.monoidal", "linear.markov"):
        return m.Diagram.contraction(x)
    if name == "feedback.cat":
        return (f >> m.Box("g", x, x)).delay()
    if name == "feedback":
        return m.Box("f", x @ x.delay(), x @ x).feedback()
    if name == "feedback.balanced":
        return m.Braid(x, x).trace().to_braided()
    if name == "feedback.symmetric":
        return m.Swap(x, x).trace(left=True)
    if name == "feedback.markov":
        return (m.Copy(x) >> x @ f).trace()
    if name == "feedback.comarkov":
        return (f @ x >> m.Merge(x)).trace(left=True)
    if name == "dataflow.symmetric":
        return m.FollowedBy(m.Ty(m.Wire("x", is_constant=False)))
    if name == "dataflow.markov":
        return (m.FollowedBy(x) >> m.Copy(x)).feedback(
            dom=x.head, cod=x, mem=x)
    if name == "closed.monoidal":
        return m.Eval(x << x)
    if name == "closed.symmetric":
        return m.Swap(x, x ** x) >> m.Eval(x ** x)
    if name == "closed.markov":
        return x ** x @ m.Copy(x) >> m.Eval(x ** x) @ x
    if name == "feedback.closed.monoidal":
        return m.Eval(x << x).trace()
    if name == "feedback.closed.balanced":
        return (m.Eval(x << x) @ x >> m.Braid(x, x)).trace()
    if name == "feedback.closed.symmetric":
        return (m.Swap(x, x ** x) >> m.Eval(x ** x)).trace(left=True)
    if name == "feedback.closed.markov":
        return (x ** x @ m.Copy(x) >> m.Eval(x ** x) @ x).trace()
    if name == "rigid":
        return x @ m.Cap(x.r, x) >> m.Cup(x, x.r) @ x
    if name == "pivotal":
        g = m.Box("f", x, m.Ty("y"))
        transpose = m.Cap(x.r, x) @ g.cod.r >> x.r @ g @ g.cod.r\
            >> x.r @ m.Cup(g.cod, g.cod.r)
        return m.Equation(transpose, g.r)
    if name == "ribbon":
        return (m.Cap(x, x.r) >> m.Braid(x, x.r)).to_ribbons()
    if name == "compact":
        return m.Cap(x, x.r) >> m.Swap(x, x.r)
    if name == "frobenius":
        return m.Spider(2, 2, x)
    raise KeyError(name)


def draw(name):
    """ Draw the picture of a module, with ``ev``, ``λ`` and ``fby`` as
    labels. """
    drawing = picture(name).to_drawing()
    for node in drawing.nodes:
        if node.kind == "box":
            node.box.drawing_name = LABELS.get(
                node.box.drawing_name, node.box.drawing_name)
    drawing.draw(path=str(PICTURES / f"{name}.svg"), figsize=(1.6, 1.2),
                 margins=(.1, .1), wire_labels=False, fontsize=10)


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
