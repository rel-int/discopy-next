"""
The categories of :mod:`discopy.abc` compiled to definitions of `Lean
<https://lean-lang.org>`_, read off the classes themselves.

Each :class:`discopy.testable.Declaration` of a class is evaluated on
:class:`Term`, symbols that record what is done to them, instead of the
morphisms of a category. A :class:`discopy.testable.Rule` gives an operation
of the structure with its typing law, e.g. ``cut`` gives ``comp`` and
``dom (comp f g) = A ∧ cod (comp f g) = C`` for ``f : A → B`` and
``g : B → C``, and an :class:`discopy.testable.Axiom` gives a law, a
proposition about these operations. A class compiles to a ``structure`` of its
operations extending those of its bases, with one ``def`` per law and a
``Laws`` predicate of all the laws it inherits. Morphisms are untyped, as in
the Python: one type ``Hom`` with ``dom`` and ``cod`` into ``Ob``, every law
assuming its morphisms have the boundary their annotation states, so that the
laws of a strict structure are plain equalities.

A category declares which of the laws it inherits are broken, see
:meth:`discopy.testable.Declaration.failing`: its :func:`theory` is the
conjunction of the others, the structure its diagrams are meant to be the
free one of. The Lean package ``discolean/`` proves it of the diagrams of
:mod:`discopy.monoidal` and checks every definition compiled here, see
:func:`generate`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Term
    Unsupported

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        declaration
        structure
        theory
        generate
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Self, get_args, get_origin

from discopy import abc, monoidal
from discopy.testable import (
    Atom, Axiom, Declaration, Equation, Hom, Inequation, Obj, Rule, Var,
    position, substitute)

LEAN_NAMES = {"then": "comp"}
""" The Lean names of methods whose own name is a keyword there. """

SORTS = {Obj: "Ob", Atom: "Ob"}
""" The Lean type of a variable, from the sort its bound states. """


class Unsupported(Exception):
    """ A declaration the compiler does not translate, with the reason. """


@dataclass(frozen=True)
class Term:
    """
    A symbol that records what is done to it: a method called on a term, or
    on the category, is a term with that method as head, the term and the
    arguments as arguments, and the sort of the result.

    Parameters:
        head : The variable, or the name of the method.
        args : The arguments of the method, terms or numbers.
        sort : ``"Ob"`` for an object, ``"Hom"`` for a morphism.
        applied : Whether the head is a method rather than a variable.

    >>> f, g = Term("f"), Term("g")
    >>> print((f >> g).dom)
    dom (comp f g)
    """
    head: str
    args: tuple = ()
    sort: str = "Hom"
    applied: bool = False

    def __str__(self):
        args = (f"({arg})" if isinstance(arg, Term) and arg.args else str(arg)
                for arg in self.args)
        return " ".join((self.name, *args))

    @property
    def name(self) -> str:
        """ The Lean name of the head: ``Ob`` after one returning objects
        from objects, as ``tensorOb`` is the tensor of objects. """
        name = LEAN_NAMES.get(self.head, self.head)
        is_object_operation = self.applied and self.sort == "Ob" and all(
            isinstance(arg, Term) and arg.sort == "Ob" for arg in self.args)
        return f"{name}Ob" if is_object_operation else name

    @property
    def signature(self) -> str:
        """ The Lean type of the head, e.g. ``Hom → Hom → Hom``. """
        return " → ".join([
            arg.sort if isinstance(arg, Term) else "Nat"
            for arg in self.args] + [self.sort])

    def operations(self) -> dict[str, str]:
        """ The signature of each operation the term applies. """
        result = {self.name: self.signature} if self.applied else {}
        for arg in self.args:
            if isinstance(arg, Term):
                result.update(arg.operations())
        return result

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("dom", "cod", "l", "r"):
            return Term(name, (self, ), "Ob" if name in ("dom", "cod")
                        else self.sort, True)
        return lambda *args, **kwargs: apply(name, (self, *args), kwargs,
                                             self.sort)

    def __matmul__(self, other):
        """
        The tensor, whiskering an object beside a morphism into its identity
        as :meth:`discopy.abc.MonoidalCategory.whisker` does.
        """
        left, right = (
            apply("id", (term, ), {}, "Hom") if term.sort == "Ob"
            and {self.sort, other.sort} == {"Ob", "Hom"} else term
            for term in (self, other))
        return left.tensor(right)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def __len__(self):
        raise Unsupported("it takes the length of a symbol")

    def __getitem__(self, key):
        raise Unsupported("it slices a symbol")


def apply(name, args, kwargs, sort) -> Term:
    """ The term of a method applied, refusing keyword arguments. """
    if kwargs:
        raise Unsupported(f"it calls {name} with keywords {sorted(kwargs)}")
    return Term(name, tuple(args), sort, True)


class Symbolic:
    """
    The category or the objects of an abstract class as symbols: calling
    one of its class methods is a term of the sort its annotation returns,
    and calling the objects themselves is the unit.
    """
    def __init__(self, cls: type):
        self.cls = cls

    def __call__(self, *args):
        return apply("unit", args, {}, "Ob")

    def equation_factory(self, *terms):
        """ The equation of the terms, unchecked. """
        return Equation(*terms)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        returns = inspect.signature(
            getattr(self.cls, name)).return_annotation
        sort = {"C0": "Ob", "C1": "Hom"}.get(
            getattr(returns, "__name__", returns))
        if sort is None:
            raise Unsupported(f"{name} returns {returns}")
        return lambda *args, **kwargs: apply(name, args, kwargs, sort)


def render(value) -> str:
    """ The Lean expression of a term, a number or a tuple of them. """
    if isinstance(value, tuple):
        return f"({', '.join(map(render, value))})"
    if isinstance(value, Term) or isinstance(value, int)\
            and not isinstance(value, bool):
        return str(value)
    raise Unsupported(f"it evaluates to {value!r}")


@dataclass
class Compiled:
    """
    A declaration evaluated on symbols, see :func:`declaration`.

    Parameters:
        binders : The Lean binders of its variables, e.g. ``(A B : Ob)``.
        premises : The boundary of each morphism it quantifies over.
        conclusion : The typing law of a rule, the equation of an axiom.
        output : What it evaluates to on the symbols.
        boundaries : The objects its premises and conclusion mention.
    """
    binders: str
    premises: list[str]
    conclusion: str
    output: object
    boundaries: tuple = ()

    def statement(self, prefix: str = "", names=()) -> str:
        """
        The Lean proposition of the declaration, its operations prefixed,
        e.g. by ``cat.`` outside the structure that has them as fields.
        """
        statement = " → ".join(self.premises + [self.conclusion])
        if self.binders:
            statement = f"∀ {self.binders}, {statement}"
        if not prefix:
            return statement
        names = set(names) | set(self.operations())
        words = statement.replace("(", "( ").replace(")", " )").split(" ")
        return " ".join(prefix + word if word in names else word
                        for word in words)\
            .replace("( ", "(").replace(" )", ")")

    def operations(self) -> dict[str, str]:
        """ The signature of each operation the declaration applies. """
        result = {}
        terms = self.output.terms if isinstance(self.output, Equation)\
            else (self.output, )
        for term in (*terms, *self.boundaries):
            for item in term if isinstance(term, tuple) else (term, ):
                if isinstance(item, Term):
                    result.update(item.operations())
        return result


def variables(cls: type, decl: Declaration) -> tuple[dict, list]:
    """
    The scope a declaration of an abstract class is evaluated in, the
    category and its objects as :class:`Symbolic`, and its binders: each
    type parameter of sort ``Obj`` or ``Atom`` an object.
    """
    category = Symbolic(cls)
    scope: dict = {Self: category}
    for klass in cls.__mro__:
        for parameter in getattr(klass, "__type_params__", ()):
            index = position(parameter, cls)
            if index is not None:
                scope[parameter] = Symbolic(cls) if index else category
    binders = []
    for parameter in inspect.unwrap(decl.function).__type_params__:
        sort = SORTS.get(get_origin(parameter.__bound__))
        if sort is None:
            raise Unsupported(f"{parameter} is not an object")
        scope[parameter] = Term(parameter.__name__, (), "Ob")
        binders.append((parameter.__name__, sort))
    return scope, binders


def conclusion(decl: Declaration, output, scope: dict) -> tuple[str, list]:
    """
    The Lean conclusion of a declaration evaluated to an output, and the
    objects it mentions: the boundary of the term a rule concludes, the
    terms of an axiom equal two by two.
    """
    if isinstance(decl, Rule):
        _, dom, cod = get_args(substitute(decl.annotations()["return"], scope))
        term = render(output)
        return (f"dom ({term}) = {render(dom)} ∧ cod ({term}) = "
                f"{render(cod)}", [dom, cod])
    if not isinstance(output, Equation) or isinstance(output, Inequation):
        raise Unsupported(f"it evaluates to {output!r}")
    terms = [render(term) for term in output.terms]
    return " ∧ ".join(
        f"{left} = {right}" for left, right in zip(terms, terms[1:])), []


def declaration(cls: type, decl: Declaration) -> Compiled:
    """
    A declaration of an abstract class evaluated on symbols, see
    :func:`variables`: a premise ``Hom[C1, A, B]`` is a morphism with its
    boundary as premises and ``Var[C0, x]`` the object ``x``.

    >>> from discopy.abc import Category
    >>> compiled = declaration(Category, Category.associativity)
    >>> print(compiled.output)
    Equation(comp (comp f g) h, comp f (comp g h))
    """
    decl = decl.bind(cls)
    scope, binders = variables(cls, decl)
    arguments, premises, boundaries = {}, [], []
    for parameter in decl.parameters:
        kind = substitute(decl.annotations()[parameter.name], scope)
        if get_origin(kind) is Var:
            arguments[parameter.name] = get_args(kind)[1]
        elif get_origin(kind) is Hom:
            _, dom, cod = get_args(kind)
            arguments[parameter.name] = Term(parameter.name)
            binders.append((parameter.name, "Hom"))
            boundaries += [dom, cod]
            premises += [f"dom {parameter.name} = {render(dom)}",
                         f"cod {parameter.name} = {render(cod)}"]
        else:
            raise Unsupported(f"it quantifies over {kind}")
    output = decl.function(scope[Self], **arguments)
    if output is NotImplemented:
        return Compiled("", [], "", output)
    statement, objects = conclusion(decl, output, scope)
    return Compiled(group(binders), premises, statement, output,
                    tuple(boundaries + objects))


def group(binders: list[tuple[str, str]]) -> str:
    """ Lean binders, one per sort, e.g. ``(A B : Ob) (f : Hom)``. """
    groups: dict[str, list[str]] = {}
    for name, sort in binders:
        groups.setdefault(sort, []).append(name)
    return " ".join(
        f"({' '.join(names)} : {sort})" for sort, names in groups.items())


def declarations(cls: type) -> dict[str, tuple[Declaration, Compiled]]:
    """
    The rules then the axioms a class states itself, evaluated on symbols,
    leaving out the axioms it declares inapplicable.
    """
    result = {}
    for kind in (Rule, Axiom):
        for name, value in vars(cls).items():
            if isinstance(value, kind):
                compiled = declaration(cls, value)
                if compiled.output is not NotImplemented:
                    result[name] = (value, compiled)
    return result


def bases(cls: type) -> list[type]:
    """ The abstract classes of :mod:`discopy.abc` a class extends. """
    return [base for base in cls.__bases__
            if base.__module__ == abc.__name__
            and issubclass(base, abc.Category)]


def operations(cls: type) -> dict[str, str]:
    """ The signature of each operation of a class and of its bases. """
    result = {"dom": "Hom → Ob", "cod": "Hom → Ob"}
    for base in bases(cls):
        result.update(operations(base))
    for _, compiled in declarations(cls).values():
        for name, signature in compiled.operations().items():
            if result.setdefault(name, signature) != signature:
                raise Unsupported(
                    f"{name} is {result[name]} and {signature}")
    return result


def summary(obj) -> str:
    """
    The first sentence of a docstring, on one line, its reStructuredText
    read as Markdown.

    >>> print(summary(abc.SpacialCategory))
    A spacial category is a `MonoidalCategory` where scalars slide: a \
scalar `s : I -> I` on the left of any morphism is the same scalar on its \
right, see `Selinger10`.
    """
    doc = " ".join((inspect.getdoc(obj) or "").split())
    doc = re.sub(r"`([^`<]*) <([^>]*)>`_", r"[\1](\2)", doc)
    doc = re.sub(r"(:[a-z:]+:)?``?([^`]*)``?", r"`\2`", doc)
    doc = doc.replace("`~", "`").replace("`t:`", "`")
    sentence = re.split(r"(?<=[a-z`)])\.\s+(?=[A-Z])", doc)[0]
    return sentence.rstrip(".") + "."


def structure(cls: type) -> str:
    """
    The Lean definitions of an abstract class: a ``structure`` of the
    operations its rules and axioms apply, extending those of its bases,
    with the typing law of each rule, then a ``def`` for each law and the
    ``Laws`` of the structure, extending those of its bases.

    Parameters:
        cls : An abstract class, e.g. :class:`discopy.abc.Category`.

    >>> print(structure(abc.DaggerCategory).splitlines()[2])
      dagger : Hom → Hom
    """
    name, parents = cls.__name__, bases(cls)
    inherited: dict[str, str] = {}
    for base in parents:
        inherited.update(operations(base))
    own = {operation: signature
           for operation, signature in operations(cls).items()
           if operation not in inherited}
    declared = declarations(cls)
    extends = ", ".join(f"{base.__name__} Ob Hom" for base in parents)
    lines = [f"/-- `{abc.__name__}.{name}`: {summary(cls)} -/",
             f"structure {name} (Ob : Type u) (Hom : Type v)"
             + (f" extends {extends}" if extends else "") + " where"]
    lines += [f"  {operation} : {signature}"
              for operation, signature in own.items()]
    for law_name, (decl, compiled) in declared.items():
        if isinstance(decl, Rule):
            lines += [f"  /-- `{law_name}`: {summary(decl)} -/",
                      f"  {law_name} : {compiled.statement()}"]
    lines.append("")
    axioms = [(law_name, decl, compiled)
              for law_name, (decl, compiled) in declared.items()
              if isinstance(decl, Axiom)]
    for law_name, decl, compiled in axioms:
        lines += [f"/-- {summary(decl)} -/",
                  f"def {name}.{law_name} (cat : {name} Ob Hom) : Prop :=",
                  f"  {compiled.statement('cat.', operations(cls))}", ""]
    lines.append(f"/-- The laws of `{name}`. -/")
    lines.append(
        f"structure {name}.Laws (cat : {name} Ob Hom) : Prop" + "".join(
            (" extends " if i == 0 else ", ")
            + f"to{base.__name__} : {base.__name__}.Laws cat.to{base.__name__}"
            for i, base in enumerate(parents))
        + (" where" if axioms else ""))
    lines += [f"  {law_name} : cat.{law_name}"
              for law_name, _, _ in axioms]
    return "\n".join(lines) + "\n"


def theory(category: type, cls: type) -> str:
    """
    The Lean proposition that a structure of an abstract class has the laws
    a category below it does not declare broken or inapplicable: the
    structure its diagrams are meant to be the free one of.

    Parameters:
        category : A concrete category, e.g. :class:`discopy.monoidal.Diagram`.
        cls : The abstract class whose operations it is stated with.

    >>> from discopy.monoidal import Diagram
    >>> print(theory(Diagram, abc.MonoidalCategory).splitlines()[-1])
      cat.unitality ∧ cat.associativity ∧ cat.tensor_unitality
    """
    laws = [
        name for klass in reversed(cls.__mro__)
        if klass.__module__ == abc.__name__ and issubclass(klass, abc.Category)
        for name, (decl, _) in declarations(klass).items()
        if isinstance(decl, Axiom) and not getattr(category, name).broken
        and getattr(category, name).applicable]
    module = category.__module__.removeprefix("discopy.")
    return "\n".join([
        f"/-- The laws of `{cls.__name__}` that `{module}."
        f"{category.__name__}` does not declare broken. -/",
        f"def {module} (cat : {cls.__name__} Ob Hom) : Prop :=",
        "  " + " ∧ ".join(f"cat.{name}" for name in laws)])


THEORIES = ((monoidal.Diagram, abc.MonoidalCategory), )
""" The theories :func:`generate` states: of each category, with a class. """


def generate() -> str:
    """
    The Lean file of every class of :mod:`discopy.abc` the compiler
    translates, in order, and of the :data:`THEORIES`, listing the classes it
    does not translate with the reason, a class whose base it does not
    translate included. Run ``python -m discopy.lean`` to print it, as
    ``discolean/DisCoLean/ABC.lean`` is.
    """
    compiled, skipped, parts = set(), [], []
    for cls in vars(abc).values():
        if not (isinstance(cls, type) and issubclass(cls, abc.Category))\
                or cls.__module__ != abc.__name__\
                or not inspect.isabstract(cls):
            continue
        missing = [base.__name__ for base in bases(cls)
                   if base not in compiled]
        try:
            if missing:
                raise Unsupported(f"it extends {', '.join(missing)}")
            parts.append(structure(cls))
            compiled.add(cls)
        except Unsupported as reason:
            skipped.append(f"- `{cls.__name__}`: {reason}")
    for category, cls in THEORIES:
        parts.append(theory(category, cls) + "\n")
    return "\n".join([
        "/-!",
        f"# The categories of `{abc.__name__}`",
        "",
        "Generated by `python -m discopy.lean` from the classes of",
        f"`{abc.__name__}`: do not edit. Each structure has the operations"
        " of its",
        "rules and axioms, untyped as in the Python, each rule its typing"
        " law and",
        "each axiom a `def`, all of them in its `Laws`. The classes it does"
        " not",
        "compile yet:",
        "", *skipped, "-/", "",
        "namespace DisCoLean.ABC", "",
        "universe u v", "",
        "variable {Ob : Type u} {Hom : Type v}", "",
        *parts, "end DisCoLean.ABC", ""])


if __name__ == "__main__":
    print(generate(), end="")
