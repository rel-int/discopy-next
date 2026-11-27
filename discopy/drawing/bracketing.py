"""
The drawings of terms, i.e. with the brackets that say how they were built.

A string diagram draws the value of a term, so that the two sides of an axiom
of :mod:`~discopy.abc` draw the same, e.g. ``(f >> g) >> h`` and
``f >> (g >> h)``. A :class:`Bracketed` term of a free category is drawn with
its :attr:`~discopy.drawing.planegraph.Drawing.bracketing`: each operation puts
every operand that is itself the result of an operation in a dotted
:meth:`~discopy.drawing.planegraph.Drawing.bracket`, a generator or an identity
in none, and the term itself, i.e. its outermost operation, in none either.
Each axiom thus draws itself from its
:meth:`~discopy.testable.Declaration.generic` instance in its
:func:`free_category`, see :func:`equation`.

A term delegates to its free category every name the axioms of that category
use and :class:`Bracketed` lacks, see :func:`delegate`: a classmethod, e.g.
``ev``, ``cups`` or ``laxator``, is a generator, drawn as its value and never
bracketed; a method or a property, e.g. ``curry``, ``bang``, ``transpose`` or
``r``, is a unary operation, which draws its operand in a dotted bracket
labelled by its symbol, followed by the arguments it is given other than its
defaults, e.g. ``λ(left=False)``:

============= ========
operation     label
============= ========
``dagger``    ``†``
``bang``      ``!``
``curry``     ``λ``
``uncurry``   ``λ⁻¹``
anything else its name
============= ========

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Bracketed

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        delegate
        label
        vocabulary
        free_category
        applies
        stated_axioms
        equation

Example
-------
>>> from discopy import markov
>>> from discopy.abc import Category
>>> equation(Category.associativity).draw(
...     doctest="docs/_static/bracketing/associativity.svg")

.. image:: /_static/bracketing/associativity.svg
    :align: center

>>> equation(markov.Diagram.copy_coassociativity).draw(
...     doctest="docs/_static/bracketing/copy-coassociativity.svg")

.. image:: /_static/bracketing/copy-coassociativity.svg
    :align: center
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module

from discopy import messages, monoidal
from discopy.abc import Category, DaggerCategory, TracedCategory
from discopy.drawing.planegraph import Drawing
from discopy.testable import Axiom, AxiomFailure, Equation, TestableGeneric
from discopy.utils import classproperty, factory, unbiased

FREE_CATEGORIES = (
    "monoidal", "spacial", "feedback", "braided", "balanced", "symmetric",
    "markov", "comarkov", "linear.monoidal", "closed.monoidal", "rigid",
    "pivotal", "ribbon", "compact", "closed.markov", "frobenius")
"""
The modules whose ``Diagram`` is a free category an axiom is drawn in, from
the most general, see :func:`free_category`.
"""

SYMBOLS = {"dagger": "†", "bang": "!", "curry": "λ", "uncurry": "λ⁻¹"}
""" The symbol of each unary operation that has one, see :func:`label`. """


def vocabulary(axiom: Axiom) -> set[str]:
    """
    The names the code of an axiom uses, e.g. those of the generators and
    the operations it states a law of.

    >>> from discopy.abc import RigidCategory
    >>> assert {"caps", "cups"} <= vocabulary(RigidCategory.snake_equations)
    """
    return set(inspect.unwrap(axiom.function).__code__.co_names)


def delegate(free: type, name: str):
    """
    The attribute of a :class:`Bracketed` term of a free category that
    delegates to the attribute of the free category with a name: a
    classmethod is a generator, a method or a property a unary operation,
    see :meth:`Bracketed.operation`, and anything else, e.g. a class, is
    that of the free category.

    Parameters:
        free : The free category.
        name : The name of the attribute.
    """
    attribute = inspect.getattr_static(free, name)
    if isinstance(attribute, classmethod | staticmethod):
        def generator(cls, *args, **kwargs):
            return cls.term(getattr(cls.free, name)(*args, **kwargs))
        return classmethod(generator)
    if isinstance(attribute, property):
        return property(lambda self: self.operation(name))
    if inspect.isfunction(attribute):
        def operation(self, *args, **kwargs):
            return self.operation(name, *args, **kwargs)
        return operation
    return attribute


def label(method, name: str, *args, **kwargs) -> str:
    """
    The label of a unary operation: its symbol, see :data:`SYMBOLS`, else its
    name, followed by the arguments it is given other than its defaults.

    Parameters:
        method : The method of the free category, taking the operand first.
        name : The name of the operation.
        args : The arguments it is given after the operand.
        kwargs : The keyword arguments it is given.

    >>> from discopy.rigid import Diagram
    >>> print(label(Diagram.curry, "curry", 1, left=False))
    λ(left=False)
    """
    signature = inspect.signature(method)
    given = signature.bind(None, *args, **kwargs).arguments
    changed = ", ".join(
        f"{key}={value!r}" for key, value in list(given.items())[1:]
        if value != signature.parameters[key].default)
    symbol = SYMBOLS.get(name, name)
    return f"{symbol}({changed})" if changed else symbol


@factory
@dataclass
class Bracketed[free](TestableGeneric, DaggerCategory, TracedCategory):
    """
    A term of the ``free`` category with its drawing, bracketed: its value is
    the arrow it evaluates to and its drawing that of the operations it was
    built with, each of which brackets its compound operands, see
    :attr:`~discopy.drawing.planegraph.Drawing.bracketing`. Its identities and
    generators delegate to the free category and its operations to both, so
    that an axiom of :mod:`~discopy.abc` evaluated on it builds the drawing of
    each of its sides.

    The operations draw as follows: :meth:`then` and :meth:`tensor` set their
    operands one after the other or side by side, :meth:`trace` and
    :meth:`feedback` draw a loop, and every other one, e.g. :meth:`dagger`,
    draws its operand in a bracket labelled by the operation, see
    :meth:`operation`, rather than drawing its value, which would draw the
    involution ``f.dagger().dagger() == f`` as ``f == f``. A term of an axiom
    is drawn on its own, its outermost operation with no bracket, see
    :meth:`to_drawing`.

    Parameters:
        value : The arrow of the free category the term evaluates to.
        drawing : The drawing of the term, that of a generator by default.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box, Diagram
    >>> x = Ty('x')
    >>> f, g = (Bracketed[Diagram](Box(name, x, x)) for name in "fg")
    >>> assert ((f >> g) >> f).value == (f >> (g >> f)).value
    >>> assert (f >> g).drawing.bracketing and not f.drawing.bracketing
    >>> (f >> g).dagger().draw(doctest="docs/_static/bracketing/dagger.svg")

    .. image:: /_static/bracketing/dagger.svg
        :align: center
    """
    value: free
    drawing: Drawing = None  # ty: ignore[invalid-assignment]

    def __post_init__(self):
        if self.drawing is None:
            self.drawing = self.value.to_drawing()

    def __class_getitem__(cls, values):
        """
        The terms of a free category, which delegate to it every name its
        axioms use that a term lacks, see :func:`delegate`.
        """
        terms, free = super().__class_getitem__(values), values
        if not isinstance(free, type):
            return terms
        names = set().union(*map(vocabulary, free.axioms.values()))
        for name in sorted(names):
            if hasattr(free, name) and not hasattr(terms, name):
                setattr(terms, name, delegate(free, name))
        return terms

    ob = classproperty(lambda cls: cls.free.ob)

    @classmethod
    def checked_params(cls) -> tuple:
        """ None: a term is drawn, the property matrix checks its value. """
        return ()

    @classmethod
    def term(cls, value):
        """
        The term of a value the free category gives: a generator if it is an
        arrow, never bracketed, else the value itself, e.g. an object.

        Parameters:
            value : The value.
        """
        # pylint: disable-next=isinstance-second-argument-not-valid-type
        return cls(value) if isinstance(value, cls.free) else value

    def operation(self, name: str, *args, **kwargs) -> Bracketed:
        """
        A unary operation of the free category on a term: its value is that
        of the operation on the value of the term, any term among the
        arguments taken as its value, and its drawing that of the term in a
        bracket labelled by the operation, see :func:`label`. A value that is
        not an arrow, e.g. an object, is returned as it is.

        Parameters:
            name : The name of the method or the property of the free
                category.
            args : The arguments of the operation after the operand.
            kwargs : The keyword arguments of the operation.

        >>> from discopy.rigid import Ty, Box, Diagram
        >>> f = Bracketed[Diagram](Box('f', Ty('x'), Ty('y')))
        >>> assert f.transpose().value == f.value.transpose()
        >>> top, _, _ = f.transpose(left=True).drawing.boxes
        >>> print(top.cod[0])
        transpose(left=True)
        """
        args = [arg.value if isinstance(arg, Bracketed) else arg
                for arg in args]
        method = inspect.getattr_static(type(self.value), name)
        if isinstance(method, property):
            value, method = getattr(self.value, name), method.fget
        else:
            value = getattr(self.value, name)(*args, **kwargs)
        # pylint: disable-next=isinstance-second-argument-not-valid-type
        if not isinstance(value, self.free):
            return value
        return type(self)(value, self.drawing.bracket(
            label(method, name, *args, **kwargs), value.dom, value.cod))

    dom = property(lambda self: self.value.dom)
    cod = property(lambda self: self.value.cod)

    def compound(self, value: Category, drawing: Drawing) -> Bracketed:
        """
        The term that a value and the drawing of an operation make, which an
        operation that takes it as an operand brackets.

        Parameters:
            value : The value of the operation.
            drawing : The drawing of the operation.
        """
        drawing = drawing.relabel_nodes()
        drawing.bracketing = True
        return type(self)(value, drawing)

    @classmethod
    def id(cls, dom=None) -> Bracketed:
        """
        The identity of the free category, drawn longer than a box so that
        composing with it shows, e.g. in
        :meth:`~discopy.abc.Category.unitality`, rather than leaving the
        drawing as it is.

        Parameters:
            dom : The object of the identity.
        """
        value = cls.free.id(dom)
        return cls(value, value.to_drawing().stretch(0.75))

    @unbiased
    def then(self, other: Bracketed) -> Bracketed:
        return self.compound(
            self.value.then(other.value), self.drawing.then(other.drawing))

    @unbiased
    def tensor(self, other: Bracketed) -> Bracketed:
        """
        The tensor of two terms, which brackets its compound operands, unless
        one of them is an identity: whiskering is drawn as no operation of its
        own, the other operand left as it is and the result as compound as it.
        """
        value = self.value.tensor(other.value)
        if not (self.drawing.is_identity or other.drawing.is_identity):
            return self.compound(
                value, self.drawing.tensor(other.drawing))
        drawing = self.plain().tensor(other.plain()).relabel_nodes()
        drawing.bracketing = (
            self.drawing.bracketing or other.drawing.bracketing)
        return type(self)(value, drawing)

    def plain(self) -> Drawing:
        """ The drawing of the term as a value, i.e. never bracketed. """
        drawing = self.drawing.relabel_nodes()
        drawing.bracketing = False
        return drawing

    def dagger(self) -> Bracketed:
        return self.operation("dagger")

    def loop(self, value: Category, left: bool = False) -> Bracketed:
        """
        The trace or the feedback of a term with a given value, drawn as a
        trace over as many wires as the value leaves out.

        Parameters:
            value : The value of the trace or the feedback.
            left : Whether the loop is on the left rather than the right.
        """
        size = len(self.cod) - len(value.cod)
        return self.compound(value, self.drawing.trace(size, left))

    def trace(self, n=None, left=False, *, dom=None, cod=None, mem=None):
        """ The trace of a term, see :meth:`loop`. """
        return self.loop(self.value.trace(
            n, left, dom=dom, cod=cod, mem=mem), left)

    def feedback(self, n=None, left=False, *, dom=None, cod=None, mem=None):
        """ The feedback of a term, see :meth:`loop`. """
        return self.loop(self.value.feedback(
            n, left, dom=dom, cod=cod, mem=mem), left)

    def delay(self, n_steps: int = 1) -> Bracketed:
        """
        The delay of a term, drawn as the delay of its value, since the delay
        of a generator is a generator.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        return type(self)(self.value.delay(n_steps))

    @classmethod
    def equation_factory(cls, *terms) -> Equation:
        """ The equation of the drawings of terms, see :meth:`to_drawing`. """
        return monoidal.Equation(*terms)

    def to_drawing(self) -> Drawing:
        """ The drawing of the term on its own, with no outer bracket. """
        return self.drawing.relabel_nodes()

    def draw(self, **params):
        """
        Draw the term, see :meth:`~discopy.drawing.planegraph.Drawing.draw`.
        """
        return self.to_drawing().draw(**params)


def free_category(axiom: Axiom) -> type | None:
    """
    The free category an axiom is drawn in: the category it is bound to if it
    is a ``Diagram``, else the first one of :data:`FREE_CATEGORIES` below the
    class stating it that does not declare it inapplicable and has every
    name it uses that one of them has, see :func:`vocabulary`, :obj:`None`
    when there is none, e.g. for :meth:`~discopy.abc.Groupoid.invertibility`.

    >>> from discopy.abc import (
    ...     Category, TracedCategory, BalancedTracedCategory)
    >>> print(free_category(Category.associativity).__module__)
    discopy.monoidal
    >>> print(free_category(TracedCategory.yanking).__module__)
    discopy.compact
    >>> print(free_category(BalancedTracedCategory.yanking).__module__)
    discopy.compact
    """
    if axiom.category is not None and axiom.category.__name__ == "Diagram":
        return axiom.category
    candidates = [
        import_module(f"discopy.{module}").Diagram
        for module in FREE_CATEGORIES]
    structure = {
        name for name in vocabulary(axiom)
        if any(hasattr(candidate, name) for candidate in candidates)}
    for candidate in candidates:
        if not issubclass(candidate, axiom.category or object):
            continue
        bound = candidate.axioms.get(axiom.name)
        if bound is not None and not applies(bound):
            continue
        if all(hasattr(candidate, name) for name in structure):
            return candidate
    return None


def applies(axiom: Axiom) -> bool:
    """
    Whether a bound axiom applies, i.e. it is not declared
    :meth:`Axiom.inapplicable`, which leaves it no argument.

    >>> from discopy.abc import Category, PivotalCategory
    >>> assert applies(Category.unitality)
    >>> assert not applies(PivotalCategory.yanking)
    """
    if axiom.parameters:
        return True
    try:
        return axiom() is not NotImplemented
    except Exception:  # pylint: disable=broad-exception-caught
        return True  # A law with no premise, which only a category builds.


def stated_axioms(category: type) -> dict[str, Axiom]:
    """
    The axioms a class of :mod:`~discopy.abc` states itself, rather than
    inherits, which :func:`equation` draws, bound to that class: those it
    declares inapplicable state nothing.

    >>> from discopy.abc import ComarkovCategory
    >>> print(*stated_axioms(ComarkovCategory))
    merge_associativity merge_commutativity merge_unitality merge_coherence
    """
    axioms = {
        name: value.bind(category) for name, value in vars(category).items()
        if isinstance(value, Axiom)}
    return {name: axiom for name, axiom in axioms.items() if applies(axiom)}


def equation(axiom: Axiom) -> Equation:
    """
    The equation an axiom states between the drawings of its terms,
    evaluated on its :meth:`Declaration.generic` instance in its
    :func:`free_category`, of which each argument becomes a
    :class:`Bracketed` term. A law the free category declares broken, e.g.
    because it is not quotiented by it, is drawn all the same, and one
    between objects, e.g. :meth:`~discopy.abc.PivotalCategory.self_dual`, as
    the equation between their identities.

    Parameters:
        axiom : The axiom, bound to a ``Diagram`` or to the class stating it.

    >>> from discopy.abc import DaggerCategory
    >>> equation(DaggerCategory.dagger_contravariance).draw(
    ...     doctest="docs/_static/bracketing/dagger-contravariance.svg")

    .. image:: /_static/bracketing/dagger-contravariance.svg
        :align: center
    """
    free = free_category(axiom)
    if free is None:
        raise ValueError(messages.NO_DRAWING.format(
            axiom, "no free category of diagrams states it."))
    axiom = axiom.bind(free)
    _, arguments = axiom.generic()
    terms = Bracketed[free]  # ty: ignore[invalid-type-form]
    arguments = {
        name: terms(argument) if isinstance(argument, free) else argument
        for name, argument in arguments.items()}
    try:
        result = axiom.bind(terms)(**arguments)
    except AxiomFailure as failure:
        result = failure.equation
    if not isinstance(result, Equation):
        raise ValueError(messages.NO_DRAWING.format(axiom, result))
    if not all(isinstance(term, terms) for term in result.terms):
        return terms.equation_factory(*map(terms.id, result.terms))
    return result
