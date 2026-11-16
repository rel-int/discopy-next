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
"""
from __future__ import annotations

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
