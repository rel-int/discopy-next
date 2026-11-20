# -*- coding: utf-8 -*-

"""
The planar linear kappa calculus, i.e. the terms of a monoidal category.

The kappa calculus of :cite:t:`Hasegawa95` is the first-order half of the
lambda calculus: a term denotes a morphism rather than an element of a
function type, so it needs no exponential and reads in any monoidal category.
A term in a context :math:`\\Gamma` of variables, each one wire, has a stack
:math:`A` of inputs still to be applied and one output :math:`b`:

.. math::

    \\Gamma_1, \\Gamma_2 \\vdash e : A \\to b
    \\qquad \\text{denotes} \\qquad
    \\Gamma_1 \\otimes A \\otimes \\Gamma_2 \\to b

- a :class:`Variable` :math:`x : b \\vdash x : I \\to b` is the identity wire
- a :class:`Constant` :math:`\\vdash f : A \\to B` is a box of the signature
- an :class:`Application` :math:`e(c)` feeds a term :math:`c` with no stack
  to the first inputs of the stack of :math:`e`, putting the context of
  :math:`c` where they were
- a :class:`Kappa` :math:`\\kappa x. e` moves a block of variables from the
  context to the stack, where they stand: it is the identity on diagrams

Terms are linear, each variable occurs once, and planar, the context is in
the order of the occurrences and nothing ever crosses, so that they evaluate
with composition and whiskering alone, see :meth:`TermBase.eval`. A statement
``let x = c in e`` is the redex :math:`(\\kappa x. e)(c)`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    TermBase
    Variable
    Constant
    Application
    Kappa
    Sampler

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        let

One output
----------

A term has exactly one output, and the translation between diagrams and
terms is faithful for the diagrams that have exactly one too:
:meth:`discopy.monoidal.Diagram.to_term` reads one ``let`` for each box and
one variable for each wire, and :meth:`TermBase.eval` gives the diagram back,
box for box. A diagram with any other number of outputs is refused rather
than read as the term of a different diagram.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, y), Box('g', x, z), Box('h', y @ z, x)
>>> diagram = f @ x >> y @ g >> h
>>> print(diagram.to_term())
(x @ x).kappa(lambda x0, x1: let(f(x0), lambda x2: \
let(g(x1), lambda x3: h(x2, x3))))
>>> assert diagram.to_term().eval() == diagram
>>> (f @ g).to_term()
Traceback (most recent call last):
    ...
ValueError: Expected exactly one output, got y @ z: a term has one.

This loses no diagram where the category has products: a
:class:`discopy.closed.Pack` box turns any closed diagram into one with
exactly one output, of :class:`discopy.closed.Product` type. A closed diagram
reads as a lambda term by default, see :meth:`discopy.closed.Diagram.to_term`,
and as a kappa term with :meth:`TermBase.from_diagram`.

>>> from discopy import closed
>>> X, Y, Z = map(closed.Ty, "XYZ")
>>> F, G = closed.Box('f', X, Y), closed.Box('g', X, Z)
>>> packed = F @ G >> closed.Pack(Y * Z)
>>> assert len(packed.cod) == 1
>>> print(TermBase.from_diagram(packed).normal_form())
(X @ X).kappa(lambda x0, x1: Pack((Y * Z))(f(x0), g(x1)))
>>> assert TermBase.from_diagram(packed).eval() == packed

Axioms
------

Beta reduction :math:`(\\kappa x. e)(c) = e[c / x]` inlines the ``let`` of a
box with one output, and :meth:`TermBase.normal_form` inlines them all. It is
sound up to interchange: the two orders of ``f @ g`` have one normal form.

>>> print(diagram.to_term().normal_form())
(x @ x).kappa(lambda x0, x1: h(f(x0), g(x1)))
>>> other = x @ g >> f @ z >> h
>>> assert other != diagram
>>> assert other.to_term().normal_form() == diagram.to_term().normal_form()

When every box has one output the normal form is a tree, and two diagrams
with one output are equal up to interchange exactly when their normal forms
are. A box with several outputs keeps its ``let``, which binds them all at
once, and two such statements do not commute: there the normal form is
sound, not complete.

>>> copy = Box('copy', x, x @ x)
>>> print((copy >> f @ g >> h).to_term().normal_form())
x.kappa(lambda x0: let(copy(x0), lambda x1, x2: h(f(x1), g(x2))))
>>> k = Box('k', x @ x @ x @ x, x)
>>> first = copy @ x >> x @ x @ copy >> k
>>> second = x @ copy >> copy @ x @ x >> k
>>> assert first.to_hypergraph() == second.to_hypergraph()
>>> assert first.to_term().normal_form() != second.to_term().normal_form()

Eta expansion :math:`\\kappa x. e(x) = e` holds on the nose.

>>> term = Constant(h)
>>> assert (y @ z).kappa(lambda a, b: term(a, b)).eval() == term.eval() == h

A term is drawn as its diagram:

>>> diagram.to_term().draw(doctest='docs/_static/kappa/let.svg')

.. image:: /_static/kappa/let.svg
    :align: center
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from inspect import signature
from itertools import count
from typing import TYPE_CHECKING, Callable, ClassVar, Iterator, Self, Sequence

from discopy import biclosed, messages
from discopy.monoidal import Box, Diagram, Ty
from discopy.testable import (
    GENERATORS, Equation, Equivalence, Serialisable, axiom)
from discopy.utils import (
    assert_isinstance, factory_key, factory_name, from_tree)

if TYPE_CHECKING:
    from hypothesis import strategies as st


class TermBase(Serialisable, Equivalence):
    """
    A term of the planar linear kappa calculus, an
    :class:`discopy.testable.Equivalence` up to the names of its bound
    variables: alpha-equivalence and the laws of it are those of
    :class:`discopy.biclosed.TermBase`, which a term of a biclosed category
    shares with its first-order half.

    Attributes:
        freevars (list[Variable]): The context, in the order of the domain.
        stack (Ty): The inputs still to be applied.
        offset (int | None): The number of free variables left of the stack,
            ``None`` when the term has no place for one: a term with no
            stack that is not a :class:`Kappa`.
        cod (Ty): The codomain, one wire unless the term is the expression
            of a :func:`let` binding several variables, or none.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, a, b = Constant(Box('f', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> term = f(a)
    >>> assert term.freevars == [a] and term.offset == 1
    >>> assert term.stack == y and term.dom == x @ y and term.cod == x
    >>> assert term(b).dom == x @ y and term(b).offset is None
    """
    freevars: list[Variable]
    stack: Ty
    offset: int | None
    cod: Ty
    serialised_attrs: ClassVar[tuple[str, ...]]

    equivalence_factory = biclosed.AlphaEquation
    alpha_eq = biclosed.TermBase.alpha_eq
    alpha_eq_under = biclosed.TermBase.alpha_eq_under
    alpha_eq_bound = staticmethod(biclosed.TermBase.alpha_eq_bound)
    choices, namings, strategy, related = (
        vars(biclosed.TermBase)[name]
        for name in ("choices", "namings", "strategy", "related"))
    alpha_renaming, alpha_soundness, alpha_completeness = (
        vars(biclosed.TermBase)[name] for name in (
            "alpha_renaming", "alpha_soundness", "alpha_completeness"))

    @property
    def parts(self) -> tuple:
        """ What the term is built from, in the order of its arguments. """
        return tuple(getattr(self, attr) for attr in self.serialised_attrs)

    def setoid(self):
        """ Terms are equal when their classes and their parts are. """
        return (factory_key(type(self)), ) + self.parts

    def __eq__(self, other):
        return isinstance(other, TermBase) and self.setoid() == other.setoid()

    def __hash__(self):
        return hash(self.setoid())

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.parts))})"

    def to_tree(self) -> dict:
        tree = lambda part: [x.to_tree() for x in part]\
            if isinstance(part, tuple)\
            else part.to_tree() if hasattr(part, "to_tree") else part
        return dict(factory=factory_name(type(self)), **{
            attr: tree(getattr(self, attr))
            for attr in self.serialised_attrs})

    @classmethod
    def from_tree(cls, tree: dict) -> Self:
        decode = lambda part: tuple(map(from_tree, part))\
            if isinstance(part, list)\
            else from_tree(part) if isinstance(part, dict) else part
        return cls(*(decode(tree[attr]) for attr in cls.serialised_attrs))

    def types(self, variables: Sequence[Variable]) -> Ty:
        """ The tensor of the types of some variables. """
        return self.cod[:0].tensor(*(x.cod for x in variables))

    @property
    def dom(self) -> Ty:
        """ The domain: the stack between the two halves of the context. """
        offset = len(self.freevars) if self.offset is None else self.offset
        return self.types(self.freevars[:offset]) @ self.stack\
            @ self.types(self.freevars[offset:])

    @property
    def constants(self) -> list[Constant]:
        """ The constants of the term, in order of occurrence. """
        return []

    @property
    def category(self) -> type[Diagram]:
        """
        The category of the boxes of the term, where it evaluates by
        default: that of :mod:`discopy.monoidal` when it has no constant.

        >>> from discopy import symmetric
        >>> f = symmetric.Box('f', symmetric.Ty('x'), symmetric.Ty('y'))
        >>> assert Constant(f).category is symmetric.Diagram
        """
        return next((
            constant.box.ar for constant in self.constants), Diagram)

    @abstractmethod
    def eval(self, functor=None):
        """
        The image of the term under a functor, the identity of its
        :attr:`category` by default, i.e. its diagram from :attr:`dom` to
        :attr:`cod`, built with composition and whiskering alone: a
        variable is an identity, a constant its box, an application
        whiskers its argument and composes, and a kappa does nothing.

        Parameters:
            functor : The functor to evaluate the types and the boxes.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', x @ y @ x, y)
        >>> term = (x @ x @ x).kappa(
        ...     lambda a, b, c: Constant(g)(a, Constant(f)(b), c))
        >>> assert term.eval() == x @ f @ x >> g
        """

    def identity(self, functor):
        """ The identity functor of the :attr:`category` by default. """
        return functor or self.category.functor_factory.id(self.category)

    def draw(self, **params):
        """ Draw the diagram of the term, see :meth:`eval`. """
        return self.eval().draw(**params)

    def __call__(self, *arguments: Term) -> Term:
        """ The term applied to each of the arguments in turn. """
        return reduce(Application, arguments, self)

    def substitute(self, substitution: dict[Variable, Term]) -> Term:
        """
        The term with terms of no stack substituted for free variables,
        each in place and renaming the bound variables it would capture.

        Parameters:
            substitution : The term to substitute for each variable.

        Example
        -------
        >>> x = Ty('x')
        >>> f, a, b = Constant(Box('f', x @ x, x)), Variable('a', x), \\
        ...     Variable('b', x)
        >>> term = x.kappa(lambda b: f(a, b))
        >>> print(term.substitute({a: b}))
        x.kappa(lambda b_: f(b, b_))
        """
        # pylint: disable=unused-argument  # a constant has no variable
        return self

    def normal_form(self) -> Term:
        """
        The beta normal form of the term: every redex
        :math:`(\\kappa x. e)(c)` with :math:`c` of the type of :math:`x`
        is contracted to :math:`e[c / x]`, in one pass since a linear
        substitution creates no redex. The term evaluates to the same
        diagram up to interchange, see :meth:`normal_form_soundness`.

        Example
        -------
        >>> x = Ty('x')
        >>> f, a = Constant(Box('f', x, x)), Variable('a', x)
        >>> assert let(f(a), lambda b: f(b)).normal_form() == f(f(a))
        """
        return self

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> Term:
        """
        The term of a diagram with one output, see
        :meth:`discopy.monoidal.Diagram.to_term`.

        Parameters:
            diagram : The diagram to read.
        """
        if len(diagram.cod) != 1:
            raise ValueError(messages.NOT_ONE_OUTPUT.format(diagram.cod))
        fresh = count()
        name = lambda typ: [Variable(f"x{next(fresh)}", typ[i:i + 1])
                            for i in range(len(typ))]
        inputs = name(diagram.dom)
        wires, steps = list(inputs), []
        for layer in diagram.inside:
            for box, offset in layer.boxes_and_offsets:
                arguments = wires[offset:offset + len(box.dom)]
                outputs = name(box.cod)
                wires[offset:offset + len(box.dom)] = outputs
                steps.append((Constant(box)(*arguments), outputs, offset))
        result, = wires
        for expression, outputs, offset in reversed(steps):
            result = expression if outputs == [result]\
                else Kappa(tuple(outputs), result, offset)(expression)
        return Kappa(tuple(inputs), result, 0) if inputs else result

    @classmethod
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def generate(cls, cod: Ty, choices: Sequence[int], types: Sequence[Ty],
                 letters: Sequence[str]) -> Term:
        """
        Build a term of codomain ``cod`` from a sequence of ``choices``, as
        :meth:`discopy.biclosed.TermBase.generate` does: its inputs are
        bound by a kappa or left free, named ``v0, v1, ...``, and the
        variable bound at each level is named by the letter at that level,
        see :class:`Sampler`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> print(TermBase.generate(y, [1, 0, 0, 2, 2, 1, 2], [x], "ab"))
        x.kappa(lambda a0: let(c0(a0), lambda b1, a2: c1(b1, a2)))
        >>> print(TermBase.generate(y, [1, 2, 1, 2, 2, 0, 2], [x], "ab"))
        let(c0, lambda a0, b1: c1(a0, b1, v0))
        """
        sampler = Sampler(iter(choices), types, letters)
        dom = [sampler.choose(types) for _ in range(sampler.choose(range(3)))]
        if sampler.choose((False, True)):
            return sampler.state(cod, tuple(
                Variable(f"v{i}", typ) for i, typ in enumerate(dom)))
        variables = tuple(
            sampler.bound(i, typ) for i, typ in enumerate(dom))
        body = sampler.state(cod, variables, len(dom))
        return Kappa(variables, body) if variables else body

    @classmethod
    def shapes(cls, *, types=None, cod=None) -> st.SearchStrategy[tuple]:
        """
        Generate the arguments of :meth:`generate` but its letters, with
        types of one wire each.
        """
        from hypothesis import strategies as st

        types = st.sampled_from(GENERATORS).map(Ty) if types is None\
            else types
        return st.tuples(
            types if cod is None else st.just(cod), cls.choices(),
            st.lists(types, min_size=1, max_size=3))

    @axiom
    def eta(cls, term: Self) -> Equation:
        """
        Eta expansion on the nose: applying a term to a variable for each
        input of its stack and abstracting them evaluates to its diagram.
        """
        variables = []
        for i in range(len(term.stack)):
            variables.append(Variable.fresh(
                "x", term.stack[i:i + 1], term, *variables))
        expanded = Kappa(
            tuple(variables), term(*variables), term.offset or 0)
        return Equation(expanded.eval(), term.eval())

    @axiom
    def normal_form_soundness(cls, term: Self) -> Equation:
        """
        Beta reduction is sound: a term and its normal form evaluate to
        diagrams with the same hypergraph, i.e. equal up to interchange.
        """
        return Equation(
            term.normal_form().eval().to_hypergraph(),
            term.eval().to_hypergraph())

    @axiom
    def normal_form_idempotence(cls, term: Self) -> Equation:
        """ A normal form is its own normal form. """
        normal_form = term.normal_form()
        return Equation(normal_form.normal_form(), normal_form)

    @axiom
    def reading(cls, term: Self) -> Equation:
        """
        Reading is a section of evaluation: the diagram of a term with one
        output reads as a term that evaluates back to it, box for box.
        """
        diagram = term.eval()
        return Equation(diagram.to_term().eval(), diagram)


class Variable(TermBase):
    """
    A variable with a name and a type of one wire, an identity diagram.

    Parameters:
        name : The name of the variable.
        cod : The type of the variable.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Variable('a', x).eval() == Diagram.id(x)
    >>> Variable('a', x @ x)
    Traceback (most recent call last):
        ...
    ValueError: Expected exactly one output, got x @ x: a term has one.
    """
    serialised_attrs = ("name", "cod")
    fresh = classmethod(biclosed.Variable.fresh.__func__)
    alpha_eq_under = biclosed.Variable.alpha_eq_under

    def __init__(self, name: str, cod: Ty):
        if len(cod) != 1:
            raise ValueError(messages.NOT_ONE_OUTPUT.format(cod))
        self.name, self.cod = name, cod
        self.freevars, self.stack, self.offset = [self], cod[:0], None

    def __str__(self):
        return self.name

    def atoms(self) -> list:
        """ A variable prints as its bare name, it is its own atom. """
        return [self]

    def eval(self, functor=None):
        functor = self.identity(functor)
        return functor.cod.id(functor(self.cod))

    def substitute(self, substitution):
        return substitution.get(self, self)


class Constant(TermBase):
    """
    A box as a term, with its domain as stack. It prints as the name of the
    box, so that a term reads as a program calling its boxes.

    Parameters:
        box : The box.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Constant(f).stack == x and Constant(f).eval() == f
    >>> print(Constant(f)(Variable('a', x)))
    f(a)
    """
    serialised_attrs = ("box", )

    def __init__(self, box: Box):
        assert_isinstance(box, Box)
        self.box, self.freevars = box, []
        self.stack, self.cod = box.dom, box.cod
        self.offset = 0 if box.dom else None

    def __str__(self):
        return self.box.name

    def atoms(self) -> list:
        """ A constant prints as the name of its box, it is its own atom. """
        return [self]

    @property
    def constants(self):
        return [self]

    def eval(self, functor=None):
        return self.identity(functor)(self.box)


class Application(TermBase):
    """
    The application of a term to an argument with no stack, which takes the
    first inputs of the stack: the context of the argument goes where they
    were, so that nothing crosses.

    Parameters:
        func : The term that is applied.
        arg : The argument, a term with no stack.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> g, a, b = Constant(Box('g', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> assert g(a, b) == Application(Application(g, a), b)
    >>> assert g(a, b).freevars == [a, b]
    >>> g(b)
    Traceback (most recent call last):
        ...
    ValueError: g of stack x @ y does not apply to b of stack Ty() and \
type y.
    """
    serialised_attrs = ("func", "arg")

    def __init__(self, func: Term, arg: Term):
        assert_isinstance(func, TermBase)
        assert_isinstance(arg, TermBase)
        if arg.stack or func.offset is None\
                or func.stack[:len(arg.cod)] != arg.cod:
            raise ValueError(messages.NOT_APPLICABLE.format(
                func, func.stack, arg, arg.stack, arg.cod))
        if set(func.freevars).intersection(arg.freevars):
            raise ValueError("Expected disjoint free variables.")
        self.func, self.arg = func, arg
        self.freevars = func.freevars[:func.offset] + arg.freevars\
            + func.freevars[func.offset:]
        self.stack, self.cod = func.stack[len(arg.cod):], func.cod
        self.offset = func.offset + len(arg.freevars) if self.stack else None

    @property
    def spine(self) -> tuple[Term, list[Term]]:
        """ The term applied first and its arguments, in order. """
        head, arguments = self.func, [self.arg]
        while isinstance(head, Application):
            head, arguments = head.func, [head.arg] + arguments
        return head, arguments

    def __str__(self):
        head, arguments = self.spine
        if isinstance(head, Kappa) and len(arguments) == 1\
                and not head.body.stack and head.stack == self.arg.cod:
            return f"let({self.arg}, {head.function})"
        return f"{head}({', '.join(map(str, arguments))})"

    def atoms(self) -> list:
        """ The atoms of an application: those of its function and arg. """
        return self.func.atoms() + self.arg.atoms()

    @property
    def constants(self):
        return self.func.constants + self.arg.constants

    def eval(self, functor=None):
        functor = self.identity(functor)
        func, identity = self.func, functor.cod.id
        left = identity(functor(self.types(func.freevars[:func.offset])))
        right = identity(functor(
            self.stack @ self.types(func.freevars[func.offset:])))
        return left @ self.arg.eval(functor) @ right >> func.eval(functor)

    def substitute(self, substitution):
        return type(self)(self.func.substitute(substitution),
                          self.arg.substitute(substitution))

    def normal_form(self):
        func, arg = self.func.normal_form(), self.arg.normal_form()
        if not isinstance(func, Kappa) or func.head is None\
                or func.head.cod != arg.cod:
            return type(self)(func, arg)
        body = func.body.substitute({func.head: arg})
        return type(func)(func.variables[1:], body) if func.variables[1:]\
            else body

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Applications with functions and arguments alike. """
        if any(type(other) is not type(self) for other in others):
            return False
        return self.func.alpha_eq_under(
            scopes, [other.func for other in others], depth)\
            and self.arg.alpha_eq_under(
                scopes, [other.arg for other in others], depth)


class Kappa(TermBase):
    """
    The abstraction of a block of ``variables`` of the ``body``, contiguous
    in its context and next to its stack if it has one: they join the stack
    where they stand, at its front if they are left of it and at its back
    otherwise, so that the kappa of a term is the same diagram.

    Parameters:
        variables : The variables to bind, in the order of the context.
        body : The term in which they are bound.
        offset : The number of free variables left of the stack, needed
            when no variable says where it is: the abstraction of no
            variable in a body with no stack, i.e. the place for a box
            with no output.

    Note
    ----
    Applying a type to a Python function with :meth:`Ty.kappa` names the
    variables after the parameters, one for each wire.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, a, b = Constant(Box('f', x @ y, x)), Variable('a', x), \\
    ...     Variable('b', y)
    >>> term = Kappa((b, ), f(a, b))
    >>> assert term.freevars == [a] and term.stack == y
    >>> assert term == y.kappa(lambda b: f(a, b)) and term.eval() == f.box
    >>> assert Kappa((a, ), term).stack == x @ y == Kappa((a, ), f(a)).stack
    >>> Kappa((b, a), f(a, b))
    Traceback (most recent call last):
        ...
    ValueError: Expected a block of variables next to the stack of f(a, b), \
got b, a.

    A box with no output is put between two wires by an ``offset``:

    >>> e, c = Constant(Box('e', x, Ty())), Variable('c', x)
    >>> term = let(e(a), lambda: f(c, b), offset=1)
    >>> assert term.freevars == [c, a, b]
    >>> assert term.eval() == x @ e.box @ y >> f.box
    """
    serialised_attrs = ("variables", "body", "offset")

    def __init__(self, variables: tuple[Variable, ...], body: Term,
                 offset: int | None = None):
        assert_isinstance(body, TermBase)
        variables = tuple(variables)
        start = body.freevars.index(variables[0])\
            if variables and variables[0] in body.freevars\
            else body.offset if offset is None else offset
        stop = (start or 0) + len(variables)
        if start is None or offset not in (None, start)\
                or tuple(body.freevars[start:stop]) != variables\
                or not 0 <= start <= len(body.freevars)\
                or body.offset not in (None, start, stop):
            raise ValueError(messages.NOT_A_BLOCK.format(
                body, ", ".join(x.name for x in variables)))
        self.variables, self.body, self.offset = variables, body, start
        self.freevars = body.freevars[:start] + body.freevars[stop:]
        self.cod, bound = body.cod, body.types(variables)
        self.stack = body.stack @ bound\
            if body.stack and body.offset == start and variables\
            else bound @ body.stack

    @property
    def head(self) -> Variable | None:
        """ The variable the first input of the stack binds, if any. """
        front = not self.body.stack or self.body.offset != self.offset
        return self.variables[0] if self.variables and front else None

    @property
    def function(self) -> str:
        """ The Python function of the body, as printed. """
        names = "".join(f" {x.name}," for x in self.variables).rstrip(",")
        offset = "" if self.variables else f", offset={self.offset}"
        return f"lambda{names}: {self.body}{offset}"

    def __str__(self):
        dom = self.body.types(self.variables)
        dom = f"({dom})" if len(dom) > 1 else str(dom)
        return f"{dom}.kappa({self.function})"

    def atoms(self) -> list:
        """
        The atoms of a kappa: those of the types of its variables and of
        its body but the variables, which the lambda it prints binds.
        """
        return self.body.types(self.variables).atoms() + [
            atom for atom in self.body.atoms() if atom not in self.variables]

    @property
    def constants(self):
        return self.body.constants

    def eval(self, functor=None):
        return self.body.eval(self.identity(functor))

    def substitute(self, substitution):
        inside = {x: term for x, term in substitution.items()
                  if x in self.freevars}
        captured = {x for term in inside.values() for x in term.freevars}
        variables, body = list(self.variables), self.body
        for i, variable in enumerate(variables):
            if variable in captured:
                variables[i] = Variable.fresh(
                    variable.name, variable.cod, body, *inside.values())
                body = body.substitute({variable: variables[i]})
        offset = self.offset + sum(
            len(inside[x].freevars) - 1
            for x in self.freevars[:self.offset] if x in inside)
        return type(self)(tuple(variables), body.substitute(inside), offset)

    def normal_form(self):
        return type(self)(
            self.variables, self.body.normal_form(), self.offset)

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Kappas of variables of the same types, with bodies alike. """
        if any(type(other) is not type(self) or other.offset != self.offset
               or [x.cod for x in other.variables]
               != [x.cod for x in self.variables] for other in others):
            return False
        terms = (self, *others)
        return self.alpha_eq_bound(
            scopes, [term.body for term in terms],
            [term.variables for term in terms], depth)

    @classmethod
    def from_callable(cls, dom: Ty, offset: int | None = None
                      ) -> Callable[[Callable], Kappa]:
        """
        The kappa of a Python function from a variable for each wire of
        ``dom``, named after its parameters, to the body.

        Parameters:
            dom : The types of the variables.
            offset : Passed to :class:`Kappa`.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Constant(Box('f', x @ x, x))
        >>> @Kappa.from_callable(x @ x)
        ... def term(a, b):
        ...     return f(a, b)
        >>> assert term == (x @ x).kappa(lambda a, b: f(a, b))
        """
        def decorator(function):
            names = list(signature(function).parameters)
            if len(names) != len(dom):
                raise ValueError(
                    f"Expected {len(dom)} variables, got {len(names)}")
            variables = tuple(Variable(name, dom[i:i + 1])
                              for i, name in enumerate(names))
            return cls(variables, function(*variables), offset)

        return decorator


type Term = Variable | Constant | Application | Kappa


def let(expression: Term, body: Callable, offset: int | None = None
        ) -> Application:
    """
    Bind the outputs of an ``expression`` inside the ``body`` of a Python
    function, i.e. the redex of its :class:`Kappa` applied to the
    expression.

    Parameters:
        expression : The term that is evaluated, with no stack.
        body : A function from a variable for each output to a term.
        offset : Passed to :class:`Kappa`, for an expression of no output.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Constant(Box('f', x, y @ y)), Constant(Box('g', y @ y, x))
    >>> term = let(f(Variable('a', x)), lambda b, c: g(b, c))
    >>> assert term == (y @ y).kappa(lambda b, c: g(b, c))(f(Variable('a', x)))
    >>> assert term.eval() == f.box >> g.box
    """
    return Kappa.from_callable(expression.cod, offset)(body)(expression)


@dataclass
class Sampler:
    """
    Samples a term with no stack from a sequence of choices, one per node,
    taking leaves once it runs out, see :meth:`TermBase.generate`.

    The term is planar and linear: each subterm is given the variables it
    uses, in order, and a node :meth:`split` them between its subterms. A
    node is a leaf, a fresh constant applied to subterms, or a statement
    binding zero, one or two outputs between the variables left for its
    body.

    Parameters:
        choices : The choices left, consumed one per node.
        types : The types of one wire the variables are drawn from.
        letters : The letters naming the bound variables, one per level.
        counter : The numbers of the constants built.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> sampler = Sampler(iter([2, 0, 1, 1, 0, 0, 0]), [x], "a")
    >>> print(sampler.state(y, (Variable('v', x), Variable('w', x))))
    let(c0(v), lambda: c1(w), offset=0)
    """
    choices: Iterator[int]
    types: Sequence[Ty]
    letters: Sequence[str]
    counter: Iterator[int] = field(default_factory=count)

    def choose(self, options: Sequence):
        """ Pick an option by the next choice, the first when they ran out. """
        return options[next(self.choices, 0) % len(options)]

    def constant(self, dom: Sequence[Ty], cod: Sequence[Ty]) -> Constant:
        """ A fresh constant between the tensors of two lists of types. """
        return Constant(Box(
            f"c{next(self.counter)}", Ty().tensor(*dom), Ty().tensor(*cod)))

    def bound(self, level: int, cod: Ty) -> Variable:
        """ The variable bound at a given level, named by its letter. """
        return Variable(
            f"{self.letters[level % len(self.letters)]}{level}", cod)

    def spine(self, cod: Sequence[Ty], bound: tuple[Variable, ...]) -> Term:
        """ A fresh constant applied to the bound variables in order. """
        return self.constant([x.cod for x in bound], cod)(*bound)

    def split(self, bound: tuple, n: int) -> list[tuple]:
        """ The bound variables cut into ``n`` contiguous blocks. """
        cuts = sorted(
            self.choose(range(len(bound) + 1)) for _ in range(n - 1))
        return [bound[i:j] for i, j in zip([0] + cuts, cuts + [len(bound)])]

    def state(self, cod: Ty, bound: tuple[Variable, ...] = (),
              level: int = 0) -> Term:
        """
        A term of a given type with no stack and the bound variables as
        context: a leaf when the choices ran out, else what the next
        choice picks.
        """
        kind = next(self.choices, 0) % 3
        if kind == 0:
            return bound[0] if [x.cod for x in bound] == [cod]\
                else self.spine([cod], bound)
        if kind == 1:
            dom = [self.choose(self.types)
                   for _ in range(1 + self.choose(range(2)))]
            return self.constant(dom, [cod])(*(
                self.state(typ, block, level)
                for typ, block in zip(dom, self.split(bound, len(dom)))))
        left, middle, right = self.split(bound, 3)
        types = [
            self.choose(self.types) for _ in range(self.choose((1, 0, 2)))]
        expression = self.state(types[0], middle, level) if len(types) == 1\
            else self.spine(types, middle)
        variables = tuple(
            self.bound(level + i, typ) for i, typ in enumerate(types))
        body = self.state(cod, left + variables + right, level + len(types))
        return Kappa(variables, body, len(left))(expression)
