"""
The drawings of terms, i.e. with the brackets that say how they were built.

A string diagram draws the value of a term, so that the two sides of an axiom
of :mod:`discopy.abc` draw the same, e.g. ``(f >> g) >> h`` and
``f >> (g >> h)``. A :class:`Bracketed` term of a free category is drawn with
its :attr:`~discopy.drawing.drawing.Drawing.bracketing`: each operation puts
every operand that is itself the result of an operation in a dotted
:meth:`~discopy.drawing.drawing.Drawing.bracket`, a generator or an identity
in none, and the term itself, i.e. its outermost operation, in none either.
Each axiom thus draws itself from its
:meth:`~discopy.testable.Declaration.generic` instance in its
:func:`free_category`, see :func:`equation`.

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

        generator
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
from discopy.drawing.drawing import Drawing
from discopy.testable import Axiom, AxiomFailure, Equation
from discopy.utils import NamedGeneric, factory, unbiased

FREE_CATEGORIES = (
    "monoidal", "spacial", "feedback", "braided", "balanced", "symmetric",
    "markov", "comarkov", "ribbon")
"""
The modules whose ``Diagram`` is a free category an axiom is drawn in, from
the most general, see :func:`free_category`.
"""

GENERATORS = (
    "swap", "braid", "twist", "copy", "discard", "merge", "noise")
"""
The generators of a free category besides its identities, which a
:class:`Bracketed` term delegates to it, see :func:`generator`.
"""


def generator(name: str) -> classmethod:
    """
    The generator of a free category with a name, as a :class:`Bracketed`
    term: the drawing of a generator is never bracketed.

    Parameters:
        name : The name of the classmethod of the free category.
    """
    def method(cls, *args, **kwargs):
        return cls(getattr(cls.free, name)(*args, **kwargs))
    method.__name__ = name
    method.__doc__ = f"The ``{name}`` of the free category, see " \
        f":meth:`discopy.abc.Category.id`."
    return classmethod(method)


@factory
@dataclass
class Bracketed[free](NamedGeneric, DaggerCategory, TracedCategory):
    """
    A term of the ``free`` category with its drawing, bracketed: its value is
    the arrow it evaluates to and its drawing that of the operations it was
    built with, each of which brackets its compound operands, see
    :attr:`discopy.drawing.drawing.Drawing.bracketing`. Its identities and
    generators delegate to the free category and its operations to both, so
    that an axiom of :mod:`discopy.abc` evaluated on it builds the drawing of
    each of its sides.

    The operations draw as follows: :meth:`then` and :meth:`tensor` set their
    operands one after the other or side by side, :meth:`trace` and
    :meth:`feedback` draw a loop, and :meth:`dagger` draws its operand in a
    bracket labelled ``†`` rather than its mirror image, which would draw the
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
        :meth:`discopy.abc.Category.unitality`, rather than leaving the
        drawing as it is.

        Parameters:
            dom : The object of the identity.
        """
        value = cls.free.id(dom)
        return cls(value, value.to_drawing().stretch(0.75))

    swap, braid, twist, copy, discard, merge, noise = map(
        generator, GENERATORS)

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
        drawing.bracketing = self.drawing.bracketing or other.drawing.bracketing
        return type(self)(value, drawing)

    def plain(self) -> Drawing:
        """ The drawing of the term as a value, i.e. never bracketed. """
        drawing = self.drawing.relabel_nodes()
        drawing.bracketing = False
        return drawing

    def dagger(self) -> Bracketed:
        return type(self)(self.value.dagger(), self.drawing.bracket(
            "†", self.drawing.cod, self.drawing.dom))

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
        """ Draw the term, see :meth:`discopy.drawing.drawing.Drawing.draw`.
        """
        return self.to_drawing().draw(**params)


def free_category(axiom: Axiom) -> type | None:
    """
    The free category an axiom is drawn in: the category it is bound to if it
    is a ``Diagram``, else the first one of :data:`FREE_CATEGORIES` below the
    class stating it that does not declare it inapplicable and has all the
    generators it uses, :obj:`None` when there is none, e.g. for
    :meth:`discopy.abc.Groupoid.invertibility`.

    >>> from discopy.abc import Category, TracedCategory, BalancedCategory
    >>> print(free_category(Category.associativity).__module__)
    discopy.monoidal
    >>> print(free_category(TracedCategory.yanking).__module__)
    discopy.symmetric
    >>> print(free_category(BalancedCategory.yanking).__module__)
    discopy.balanced
    """
    if axiom.category is not None and axiom.category.__name__ == "Diagram":
        return axiom.category
    names = set(inspect.unwrap(axiom.function).__code__.co_names)
    for module in FREE_CATEGORIES:
        candidate = import_module(f"discopy.{module}").Diagram
        if not issubclass(candidate, axiom.category or object):
            continue
        bound = candidate.axioms.get(axiom.name)
        if bound is not None and not applies(bound):
            continue
        if all(hasattr(candidate, name)
               for name in names.intersection(GENERATORS)):
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
    return bool(axiom.parameters) or axiom() is not NotImplemented


def stated_axioms(category: type) -> dict[str, Axiom]:
    """
    The axioms a class of :mod:`discopy.abc` states itself, rather than
    inherits, which :func:`equation` draws, bound to that class: those it
    declares inapplicable state nothing.

    >>> from discopy.abc import MarkovCategory
    >>> print(*stated_axioms(MarkovCategory))
    copy_coassociativity copy_cocommutativity copy_counitality copy_coherence
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
    because it is not quotiented by it, is drawn all the same.

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
    return result
