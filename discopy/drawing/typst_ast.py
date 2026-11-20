"""
Typst AST for generating CeTZ/Fletcher string diagrams.

Example output::

    #import "@preview/cetz:0.5.2": canvas, draw

    #canvas({
        import draw: *
        bezier((0, 0), (1, 0), (0.3, -0.5), (0.7, 0.5))
        line((0.6, 0.5), (1.8, 0.5), (1.8, 1.1), (0.6, 1.1), close: true)
        content((1.2, 0.8), [$f$])
        content((1.2, 0.2), "a name")
    })

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    TypstNode
    Str
    Float
    Bool
    RawText
    Ident
    Coord
    Call
    ContentExpr
    Import
    Canvas
    Document

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        latex_math
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class TypstNode:
    """Base class for Typst AST nodes."""
    def render(self, indent: int = 0) -> str:
        """ The Typst source of a node, at a level of indentation. """
        raise NotImplementedError


@dataclass
class Str(TypstNode):
    """
    A quoted string literal, its backslashes and quotes escaped.

    >>> print(Str('say "hi"').render())
    "say \\"hi\\""
    """
    value: str

    def render(self, indent=0):
        escaped = self.value
        for character, escape in (
                ("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"),
                ("\r", "\\r"), ("\t", "\\t")):
            escaped = escaped.replace(character, escape)
        return f'"{escaped}"'


@dataclass
class Float(TypstNode):
    """
    A floating-point literal, rounded to ``prec`` decimal places with the
    trailing zeros of its fraction dropped.

    >>> Float(0.5).render(), Float(2.0).render(), Float(10, prec=0).render()
    ('0.5', '2', '10')
    """
    value: float
    prec: int = 4

    def render(self, indent=0):
        s = f"{round(self.value, self.prec):.{self.prec}f}"
        return s.rstrip('0').rstrip('.') if '.' in s else s


@dataclass
class Bool(TypstNode):
    """A boolean literal: ``true`` or ``false``."""
    value: bool

    def render(self, indent=0):
        return "true" if self.value else "false"


@dataclass
class RawText(TypstNode):
    """Raw Typst code inserted verbatim into the output."""
    text: str

    def render(self, indent=0):
        pre = " " * (indent * 4)
        return pre + self.text


@dataclass
class Ident(TypstNode):
    """A Typst identifier, e.g. ``bezier``, ``draw``, ``canvas``."""
    name: str

    def render(self, indent=0):
        return self.name


@dataclass
class Coord(TypstNode):
    """A 2D coordinate: ``(x, y)``."""
    x: float
    y: float
    prec: int = 4

    def render(self, indent=0):
        x = Float(self.x, self.prec).render()
        y = Float(self.y, self.prec).render()
        return f"({x}, {y})"


@dataclass
class Call(TypstNode):
    """A function call: ``func(arg1, arg2, kw=val)``."""
    func: TypstNode
    args: list[TypstNode] = field(default_factory=list)
    kwargs: dict[str, TypstNode] = field(default_factory=dict)

    def render(self, indent=0):
        pre = " " * (indent * 4)
        parts = [a.render() for a in self.args]
        parts += [f"{k}: {v.render()}" for k, v in self.kwargs.items()]
        inner = ", ".join(parts)
        return f"{pre}{self.func.render()}({inner})"


@dataclass
class ContentExpr(TypstNode):
    """
    Typst content: ``[$math$]``, or plain text as a string literal, so that
    no character of a name is read as markup.

    >>> ContentExpr("x^2", math=True).render(), ContentExpr("f]#x").render()
    ('[$x^2$]', '"f]#x"')
    """
    text: str
    math: bool = False

    def render(self, indent=0):
        pre = " " * (indent * 4)
        return pre + (
            f"[${self.text}$]" if self.math else Str(self.text).render())


@dataclass
class Import(TypstNode):
    """A ``#import`` statement."""
    path: str
    members: list[str] = field(default_factory=list)
    alias: str | None = None

    def render(self, indent=0):
        path = Str(self.path).render()
        if self.members and self.alias:
            raise ValueError("An import has members or an alias, not both.")
        if self.members:
            return f'#import {path}: {", ".join(self.members)}'
        if self.alias:
            return f'#import {path} as {self.alias}'
        return f'#import {path}'


@dataclass
class Canvas(TypstNode):
    """A ``cetz.canvas({ ... })`` block."""
    canvas_kwargs: dict[str, TypstNode] = field(default_factory=dict)
    body: list[TypstNode] = field(default_factory=list)

    def render(self, indent=0):
        pre = " " * (indent * 4)
        kw_parts = [
            f"{k}: {v.render()}"
            for k, v in self.canvas_kwargs.items()]
        kw_str = (", ".join(kw_parts) + ", ") if kw_parts else ""
        body_lines = []
        for stmt in self.body:
            rendered = stmt.render(indent + 1)
            if not rendered.endswith("\n"):
                rendered += "\n"
            body_lines.append(rendered)
        inner = "".join(body_lines)
        return f"{pre}#canvas({kw_str}{{\n{inner}{pre}}})"


@dataclass
class Document(TypstNode):
    """Top-level Typst document with imports and a canvas."""
    imports: list[Import] = field(default_factory=list)
    preamble: list[TypstNode] = field(default_factory=list)
    content: TypstNode | None = None

    def render(self, indent=0):
        parts = [i.render(indent) + "\n" for i in self.imports]
        for node in self.preamble:
            parts.append(node.render(indent) + "\n")
        if self.content:
            parts.append("\n")
            parts.append(self.content.render(indent))
            parts.append("\n")
        return "".join(parts)


GREEK = [
    "alpha", "beta", "gamma", "delta", "zeta", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda",
    "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]

LATEX_SYMBOLS = {name: name for name in GREEK} | {
    "epsilon": "epsilon.alt", "varepsilon": "epsilon", "phi": "phi.alt",
    "varphi": "phi", "vartheta": "theta.alt", "varpi": "pi.alt",
    "varrho": "rho.alt", "varsigma": "sigma.alt", "varkappa": "kappa.alt",
    "to": "arrow.r", "rightarrow": "arrow.r", "gets": "arrow.l",
    "leftarrow": "arrow.l", "leftrightarrow": "arrow.l.r",
    "Rightarrow": "arrow.r.double", "Leftarrow": "arrow.l.double",
    "Leftrightarrow": "arrow.l.r.double", "iff": "arrow.l.r.double.long",
    "implies": "arrow.r.double.long", "longrightarrow": "arrow.r.long",
    "longleftarrow": "arrow.l.long", "Longrightarrow": "arrow.r.double.long",
    "Longleftarrow": "arrow.l.double.long", "mapsto": "arrow.r.bar",
    "mapsfrom": "arrow.l.bar", "longmapsto": "arrow.r.long.bar",
    "uparrow": "arrow.t", "downarrow": "arrow.b",
    "hookrightarrow": "arrow.r.hook", "rightsquigarrow": "arrow.r.squiggly",
    "leq": "lt.eq", "le": "lt.eq", "geq": "gt.eq", "ge": "gt.eq",
    "neq": "eq.not", "ne": "eq.not", "sim": "tilde.op", "simeq": "tilde.eq",
    "cong": "tilde.equiv", "approx": "approx", "equiv": "equiv",
    "in": "in", "notin": "in.not", "ni": "in.rev", "subset": "subset",
    "subseteq": "subset.eq", "supset": "supset", "supseteq": "supset.eq",
    "models": "models", "vdash": "tack.r", "dashv": "tack.l",
    "prec": "prec", "succ": "succ", "propto": "prop", "perp": "perp",
    "mid": "divides", "otimes": "times.o", "oplus": "plus.o",
    "times": "times", "cdot": "dot.op", "circ": "compose",
    "bullet": "bullet", "star": "star.op", "ast": "ast.op",
    "pm": "plus.minus", "mp": "minus.plus", "cup": "union", "cap": "inter",
    "wedge": "and", "land": "and", "vee": "or", "lor": "or", "neg": "not",
    "lnot": "not", "setminus": "without", "dagger": "dagger",
    "ddagger": "dagger.double", "div": "div", "sum": "sum",
    "prod": "product", "coprod": "product.co", "int": "integral",
    "oint": "integral.cont", "bigotimes": "times.o.big",
    "bigoplus": "plus.o.big", "bigcup": "union.big", "bigcap": "inter.big",
    "infty": "infinity", "partial": "partial", "nabla": "nabla",
    "forall": "forall", "exists": "exists", "emptyset": "emptyset",
    "varnothing": "emptyset", "ell": "ell", "top": "top", "bot": "bot",
    "ldots": "dots.h", "dots": "dots.h", "cdots": "dots.c",
    "vdots": "dots.v", "ddots": "dots.down", "langle": "chevron.l",
    "rangle": "chevron.r", "vert": "bar.v", "Vert": "bar.v.double",
    "colon": "colon", "prime": "prime", "aleph": "aleph",
    "sin": "sin", "cos": "cos", "tan": "tan", "log": "log", "ln": "ln",
    "exp": "exp", "lim": "lim", "max": "max", "min": "min", "det": "det",
    "dim": "dim", "ker": "ker", "sup": "sup", "inf": "inf",
    ",": "thin", ":": "med", ">": "med", ";": "thick", " ": "space",
    "quad": "quad", "qquad": "wide", "!": "", "left": "", "right": "",
    "{": "\\{", "}": "\\}", "_": "\\_", "$": "\\$", "#": "\\#",
    "%": "\\%", "&": "\\&"}

LATEX_FUNCTIONS = {
    "mathbb": "bb", "mathbf": "bold", "boldsymbol": "bold",
    "mathrm": "upright", "mathit": "italic", "mathcal": "cal",
    "mathfrak": "frak", "mathsf": "sans", "mathtt": "mono", "sqrt": "sqrt",
    "hat": "hat", "widehat": "hat", "tilde": "tilde", "widetilde": "tilde",
    "bar": "overline", "overline": "overline", "underline": "underline",
    "vec": "arrow", "dot": "dot", "ddot": "dot.double",
    "frac": "frac", "dfrac": "frac", "tfrac": "frac", "binom": "binom"}
""" The LaTeX commands :func:`latex_math` reads as a Typst function, of
two arguments for a fraction and a binomial, of one otherwise. """

LATEX_TEXT = {"text": "", "textrm": "", "mbox": "", "operatorname": "op"}
""" The LaTeX commands of a text in braces, written as a Typst string in the
Typst function they map to, if any. """

TEXT_ESCAPES = {"\\" + char for char in "%$#&{}_"}
""" The escapes of LaTeX in a text, read as the character they escape. """

UNESCAPED = set("-+=")
""" The characters :func:`latex_math` leaves as they are, every other one
that is not a letter or a digit being escaped. """

LATEX_TOKEN = re.compile(
    r"\\[A-Za-z]+|\\.|'+|\s+|.", re.DOTALL)

NUMBER = re.compile(r"\d+\.?\d*")


def latex_math(source: str) -> str:
    """
    The Typst mathematics of the LaTeX mathematics ``source``, i.e. what is
    between the dollars of a label, with :data:`LATEX_SYMBOLS`,
    :data:`LATEX_FUNCTIONS` and the text commands of :data:`LATEX_TEXT`.

    A letter is its own variable, as LaTeX reads it, and every other
    character is escaped, so that e.g. ``<=`` is two symbols as in LaTeX and
    not Typst's shorthand for ``≤``.

    Raises
    ------
    ValueError
        On a command it does not know, rather than draw something else.

    Example
    -------
    >>> print(latex_math(r"\\mapsto"))
    arrow.r.bar
    >>> print(latex_math(r"x_1 <= \\frac{\\alpha}{2}"))
    x_(1) \\< = frac(alpha, 2)
    >>> print(latex_math(r"\\mathbb{R}^n \\text{ for all } n"))
    bb(R)^(n) " for all " n
    >>> latex_math(r"\\foo")
    Traceback (most recent call last):
    ...
    ValueError: No Typst for the LaTeX command \\foo in '\\\\foo'.
    """
    tokens = LATEX_TOKEN.findall(source)[::-1]
    atoms = latex_atoms(tokens, source)
    if tokens:
        raise ValueError(f"Unbalanced braces in {source!r}.")
    return " ".join(atoms)


def latex_atoms(tokens: list[str], source: str, closing=None) -> list[str]:
    """ The Typst atoms of the tokens of ``source`` popped from the end of
    ``tokens`` up to ``closing``, which is popped too. """
    atoms: list[str] = []
    while tokens and tokens[-1] != closing:
        token = tokens.pop()
        if token in "^_":
            base = atoms.pop() if atoms else '""'
            atoms.append(f"{base}{token}({latex_argument(tokens, source)})")
        elif token.startswith("'"):
            atoms.append((atoms.pop() if atoms else '""') + token)
        elif token in "{}":
            if token == "}":
                raise ValueError(f"Unbalanced braces in {source!r}.")
            atoms.append(latex_group(latex_atoms(tokens, source, "}")))
        elif token.startswith("\\"):
            atoms.append(latex_command(token[1:], tokens, source))
        elif atoms and NUMBER.fullmatch(atoms[-1]) and (token.isdigit() or (
                token == "." and tokens and tokens[-1].isdigit())):
            atoms[-1] += token
        elif token.isalnum() or token in UNESCAPED:
            atoms.append(token)
        elif not token.isspace():
            atoms.append("\\" + token)
    if closing is not None:
        if not tokens:
            raise ValueError(f"Unbalanced braces in {source!r}.")
        tokens.pop()
    return [atom for atom in atoms if atom]


def latex_group(atoms: list[str]) -> str:
    """ The Typst of a group in braces: one atom for a script to apply to,
    in an ``lr`` with no delimiters when it has several. """
    return f"lr({' '.join(atoms)})" if len(atoms) > 1 else "".join(atoms)


def skip_space(tokens: list[str]) -> list[str]:
    """ Pop the space at the end of ``tokens``, which LaTeX ignores in
    mathematics, and return them. """
    while tokens and tokens[-1].isspace():
        tokens.pop()
    return tokens


def latex_argument(tokens: list[str], source: str) -> str:
    """ The Typst of the argument of a command or a script: a group in
    braces or else the next token, a single digit for a number. """
    if not skip_space(tokens):
        raise ValueError(f"Missing an argument in {source!r}.")
    if tokens[-1] == "{":
        tokens.pop()
        return " ".join(latex_atoms(tokens, source, closing="}"))
    token = tokens.pop()
    if token.startswith("\\"):
        return latex_command(token[1:], tokens, source)
    return " ".join(latex_atoms([token], source))


def latex_command(name: str, tokens: list[str], source: str) -> str:
    """ The Typst of the LaTeX command ``\\name``, its arguments popped
    from ``tokens``. """
    if name in LATEX_TEXT:
        text, function = latex_text(tokens, source), LATEX_TEXT[name]
        return f"{function}({text})" if function else text
    if name == "sqrt" and skip_space(tokens) and tokens[-1] == "[":
        tokens.pop()
        index = " ".join(latex_atoms(tokens, source, closing="]"))
        return f"root({index}, {latex_argument(tokens, source)})"
    if name in LATEX_FUNCTIONS:
        arity = 2 if LATEX_FUNCTIONS[name] in ("frac", "binom") else 1
        arguments = [latex_argument(tokens, source) for _ in range(arity)]
        return f"{LATEX_FUNCTIONS[name]}({', '.join(arguments)})"
    if name not in LATEX_SYMBOLS:
        raise ValueError(
            f"No Typst for the LaTeX command \\{name} in {source!r}.")
    if name in ("left", "right") and skip_space(tokens) and tokens[-1] == ".":
        tokens.pop()
    return LATEX_SYMBOLS[name]


def latex_text(tokens: list[str], source: str) -> str:
    """ The text in braces after a command such as ``\\text``, as a Typst
    string. """
    if not skip_space(tokens) or tokens.pop() != "{":
        raise ValueError(f"Missing the braces of a text in {source!r}.")
    depth, text = 0, ""
    while tokens and (depth or tokens[-1] != "}"):
        token = tokens.pop()
        depth += (token == "{") - (token == "}")
        text += token[1:] if token in TEXT_ESCAPES else token
    if not tokens:
        raise ValueError(f"Unbalanced braces in {source!r}.")
    tokens.pop()
    return Str(text).render()
