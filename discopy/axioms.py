"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_: an :class:`Axiom` is stated once
on an abstract base class of :mod:`discopy.abc` as a sequent — see
:mod:`discopy.pattern` for the language of the rules — every subclass
inherits it, and a type generates the terms it quantifies over through
:meth:`Testable.strategy`, by :meth:`discopy.axioms.Testable.sample` for
diagrams. :meth:`Testable.matrix` lists every law of every type that does,
which ``proptest/`` checks against generated terms.

A law a type breaks is declared :meth:`Axiom.failing` where it breaks,
one that does not apply :meth:`Axiom.inapplicable`, and one that holds
up to a quotient or on a subspace :meth:`Axiom.modulo` and
:meth:`Axiom.weaken`. :meth:`Axiom.falsify` searches for a shrunk
counterexample on demand, and :meth:`Axiom.draw` draws a law as the
equation, or the inequation, it is on its canonical instance.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Equation
    Axiom
    AxiomFailure
    Testable
    Serialisable

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        axiom
"""


import inspect
import pickle
import sys
from abc import ABCMeta
from collections.abc import Callable
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, replace
from functools import wraps
from operator import attrgetter
from typing import TYPE_CHECKING, ClassVar, Self

from discopy.pattern import (
    Count, Declaration, Hom, Obj, Pattern, Rule, Substitution)
from discopy.utils import (
    AxiomError,
    NamedGeneric,
    classproperty,
    factory_name,
)

if TYPE_CHECKING:
    from hypothesis import strategies as st


class Equation[ar](NamedGeneric):
    """
    An equation is a list of ``terms`` to be compared up to a function
    ``up_to``, the identity by default.  Casting it to ``bool`` checks
    whether its terms are all equal up to that function.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between each pair of terms, ``"="`` by default.
        symbols : The symbols between each pair of consecutive terms, one
            fewer than the terms, overriding ``symbol``.
        up_to : The function up to which ``bool(equation)`` compares its
            terms, overriding the subclass' :attr:`up_to` if given.

    Example
    -------
    The number of boxes inside an arrow is left unchanged by associativity,
    so we can compare arrows up to the function that counts them modulo 2:

    >>> from discopy.cat import Ob, Box, Equation
    >>> x = Ob('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> parity = lambda term: len(term.inside) % 2
    >>> assert not Equation(f, f >> g >> g)
    >>> assert Equation(f, f >> g >> g, up_to=parity)
    """
    up_to = None

    def __init__(self, *terms, symbol="=", symbols=None, up_to=None):
        self.terms = terms
        gaps = max(len(terms) - 1, 0)
        self.symbols = gaps * (symbol, ) if symbols is None\
            else tuple(symbols)
        if len(self.symbols) != gaps:
            raise ValueError(
                f"Expected {gaps} symbols between {len(terms)} terms, "
                f"got {len(self.symbols)}.")
        if up_to is not None:
            self.up_to = up_to

    def modulo(self, up_to: Callable) -> Equation:
        """
        The same equation compared up to the given function, rebinding
        :attr:`up_to`, whose name the attribute already takes.

        >>> from discopy.cat import Ob, Box, Equation
        >>> x = Ob('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> assert Equation(f >> g, g >> f).modulo(lambda _: True)
        """
        return type(self)(*self.terms, symbols=self.symbols, up_to=up_to)

    def __repr__(self):
        """
        >>> from discopy.cat import Ob, Box, Equation
        >>> Equation(Box('f', Ob('x'), Ob('x')))
        cat.Equation(cat.Box('f', cat.Ob('x'), cat.Ob('x')))
        """
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f"Equation({', '.join(map(str, self.terms))})"

    def __bool__(self):
        terms = self.terms if self.up_to is None\
            else list(map(self.up_to, self.terms))
        return all(term == terms[0] for term in terms)

    def checked(self) -> Equation:
        """
        The same equation with ``≠`` between two consecutive terms that
        differ up to :attr:`up_to`, i.e. the inequation it is when it fails.

        >>> from discopy.cat import Ob, Box, Equation
        >>> x = Ob('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> Equation(f, f, g).checked().symbols
        ('=', '$\\\\neq$')
        """
        symbols = (
            symbol if type(self)(left, right, up_to=self.up_to)
            else "$\\neq$" for symbol, left, right
            in zip(self.symbols, self.terms, self.terms[1:]))
        return type(self)(*self.terms, symbols=symbols, up_to=self.up_to)


class AxiomFailure(AxiomError):
    """
    A law declared broken, raised when the bound axiom is called: the
    reason is the message and :attr:`equation` is the law evaluated on the
    arguments, whose sides say how it failed.
    """

    def __init__(self, reason: str, equation):
        super().__init__(reason, equation)
        self.equation = equation


class Testable[T](metaclass=ABCMeta):
    """
    A testable class states axioms, which its subclasses inherit along
    with the structure they axiomatise, and says how to generate the
    terms those axioms quantify over.

    Both kinds of law meet here: a :class:`discopy.abc.Category` states
    those of a categorical structure, a :class:`Serialisable` those of
    writing a term down and reading it back. A class need not be a
    category to state laws, which is why the two meet here rather than in
    either of them.

    A type that implements :meth:`strategy` is one the property matrix
    in ``proptest/`` quantifies over, checking each of its axioms against
    generated terms. One that does not is not checked, and says so by
    leaving :meth:`strategy` to raise.
    """

    max_depth: ClassVar[int] = 3
    """ The number of rules a term :meth:`sample` builds may nest. """

    @classmethod
    def strategy(cls, pattern: Pattern[T] | None = None
                 ) -> "st.SearchStrategy[T]":
        """
        Build a `search strategy
        <https://hypothesis.readthedocs.io/en/latest/data.html>`_ for the
        instances of ``cls`` a ground pattern stands for, e.g.
        ``Hom(x, y)`` the morphisms from ``x`` to ``y`` and ``Obj(size=2)``
        the objects of size two, or any instance when the pattern is
        :obj:`None`, the default. Implementing it is how a class enrols
        itself in the property matrix.

        The default raises: a class states its laws as soon as it has
        them, and is checked against them once it says how to sample their
        terms. It is deliberately not an :func:`abc.abstractmethod`,
        which would make every category that has not implemented one
        uninstantiable rather than merely unchecked.

        >>> from discopy.monoidal import Layer
        >>> Layer.strategy()
        Traceback (most recent call last):
         ...
        NotImplementedError: No search strategy implemented for Layer
        """
        raise NotImplementedError(
            f"No search strategy implemented for {cls.__name__}")

    @classproperty
    def axioms(cls: type) -> dict[str, Axiom]:
        """
        The axioms inherited by ``cls``, by name, subclasses overriding
        bases: assigning anything that is not an axiom over an inherited
        one drops it altogether, rather than restating it.
        """
        return Axiom.inherited(cls)

    @classmethod
    def sample(cls, goal: Pattern | Declaration | None = None
               ) -> "st.SearchStrategy":
        """
        Sample at random by applying the rules of the category: a term
        the ``goal`` stands for, a ground :class:`discopy.pattern.Hom`,
        any term for anything else, or the arguments of a declaration
        bound to the category.

        A term is a free :attr:`Box` or the conclusion of a rule chosen at
        random among those whose conclusion matches the goal, its premises
        sampled in turn: a morphism of the category as a term while fewer
        than :attr:`max_depth` rules are nested, anything else by the
        strategy of its class on the ground pattern of the premise. A
        term built outside its goal is a rule lying about its
        conclusion, an :class:`discopy.utils.AxiomError`, and an example
        whose residual equations fail is rejected. The premises of an
        :class:`Axiom` keep to the subspace it was weakened to.

        >>> from hypothesis import find
        >>> from discopy.monoidal import Ty, Diagram
        >>> x, y = Ty('x'), Ty('y')
        >>> term = find(Diagram.sample(Hom(x, y)),
        ...             lambda term: len(term.boxes) > 1)
        >>> assert (term.dom, term.cod) == (x, y)
        """
        from hypothesis import assume, strategies as st

        rules = list(Rule.inherited(cls).values())

        @st.composite
        def arguments(draw, declaration, subst, depth):
            ob = getattr(declaration.category or cls, "ob", None)
            subst = Substitution(subst, residuals=subst.residuals)
            subspace = declaration.subspace\
                if isinstance(declaration, Axiom) else None

            def value(pattern, label):
                ground = subst.instantiate(pattern, ob)
                if isinstance(ground, Count):
                    return draw(st.integers(0, Count.maximum), label=label)
                if not isinstance(ground, (Obj, Hom)):
                    return ground
                resolved = declaration.resolve(ground)
                ground = replace(ground, head=None)
                if isinstance(declaration, Rule) and resolved is cls\
                        and isinstance(ground, Hom):
                    return draw(term(ground, depth - 1), label=label)
                strategy = resolved.strategy(ground)
                if subspace is not None and isinstance(ground, Hom):
                    strategy = strategy.filter(attrgetter(subspace))
                return draw(strategy, label=label)

            variables = sorted(declaration.variables.items(), key=lambda
                               item: not isinstance(item[1], Count))
            for label, sort in variables:
                if label not in subst:
                    subst[label] = value(sort, label)
            for pattern, residual in subst.residuals:
                assume(subst.instantiate(pattern, ob) == residual)
            return {label: value(premise, label)
                    for label, premise in declaration.premises.items()}

        @st.composite
        def term(draw, goal, depth):
            options = [None] + [
                (rule, found) for rule in rules
                if depth > 0 or not rule.recursive
                for found in [list(rule.match(goal))] if found]
            choice = draw(st.sampled_from(options))
            if choice is None:
                result = draw(cls.Box.strategy(goal))
            else:
                rule, found = choice
                result = rule.apply(draw(arguments(
                    rule, draw(st.sampled_from(found)), depth)))
            if goal.dom not in (None, result.dom)\
                    or goal.cod not in (None, result.cod):
                raise AxiomError(
                    f"{choice[0] if choice else 'A free box'} concludes "
                    f"{goal.dom} -> {goal.cod} but built "
                    f"{result.dom} -> {result.cod}.")
            return result

        if isinstance(goal, Declaration):
            return arguments(goal, Substitution(), cls.max_depth)
        return term(goal if isinstance(goal, Hom) else Hom(), cls.max_depth)

    @classmethod
    def enrolled(cls) -> tuple[type[Testable], ...]:
        """
        The subclasses that state laws and generate the terms those laws
        quantify over, by name: a type enrols itself by implementing
        :meth:`strategy`.

        >>> from discopy.cat import Arrow
        >>> assert Arrow in Testable.enrolled()
        """
        def generates(testable):
            try:
                testable.strategy()
            except NotImplementedError:
                return False
            return True

        return tuple(sorted(
            (testable for testable in cls.subclasses()
             if testable.axioms and generates(testable)),
            key=factory_name))

    @classmethod
    def matrix(cls, once: bool = False) -> list[Axiom]:
        """
        Every law of every :meth:`enrolled` type, bound to it, or only
        ``once`` per declaration of a law: bound to the enrolled type
        nearest the class declaring it, rather than to every type
        inheriting it. A type restating a law, broken, weakened or modulo
        a quotient, declares it anew and is the nearest to that.

        >>> from discopy.cat import Arrow
        >>> assert Arrow.unitality in Testable.matrix(once=True)
        """
        cells = [
            law for testable in cls.enrolled()
            for law in testable.axioms.values()]
        if not once:
            return cells
        nearest: dict = {}
        for law in cells:
            mro = law.bound.__mro__
            distance, declaring = next(
                (i, base) for i, base in enumerate(mro)
                if law.name in vars(base))
            key = (declaring, law.name)
            if key not in nearest or distance < nearest[key][0]:
                nearest[key] = (distance, law)
        return sorted(
            (law for _, law in nearest.values()),
            key=lambda law: (factory_name(law.bound), law.name))

    @classmethod
    def subclasses(cls) -> tuple[type[Testable], ...]:
        """
        Every transitive subclass of ``cls``, ``cls`` itself included.

        A subclass is listed once, however many paths reach it, in the
        order a breadth-first walk of the subclass graph first meets
        it.

        Example
        -------
        >>> from discopy.cat import Arrow, Box
        >>> assert Arrow.subclasses()[0] is Arrow
        >>> assert Box in Arrow.subclasses()  # a subclass of a subclass
        """
        found, queue = {cls: None}, [cls]
        while queue:
            for subclass in queue.pop(0).__subclasses__():
                if subclass not in found:
                    found[subclass] = None
                    queue.append(subclass)
        return tuple(found)


no_strategy = Testable.__dict__["strategy"]
"""
The default :meth:`Testable.strategy`, under a name that can be assigned.

A class inherits it from :class:`Testable` unless a base it refines
implements one: the terms of a :class:`discopy.monoidal.Ty` are not
those of the :class:`discopy.cat.Ob` it subclasses, so a class that
would inherit the wrong strategy declares ``strategy = no_strategy``
until it implements its own.
"""


@dataclass(repr=False)
class Axiom[**P, T](Declaration[P, T]):
    """
    An axiom of a category: a :class:`discopy.pattern.Declaration` with no
    conclusion, whose premises are the arguments of a property test. The
    axiom is a classmethod of the category it is bound to, implicitly:
    its first parameter is the category and the remaining ones are
    generated from their patterns.

    Calling a bound axiom returns its verdict: :obj:`NotImplemented` when
    the structure does not apply to the category, the equation itself
    otherwise; a law declared broken raises an :class:`AxiomFailure`
    carrying that equation instead. A law is broken when *some* argument
    is a counterexample, so :attr:`broken` is declared by :meth:`failing`
    before any argument is generated: the property matrix marks such a
    law as an expected failure and lets sampling find the counterexample.

    Parameters:
        function : The function stating the law, from the category and the
            generated arguments to an :class:`Equation`, or to
            :obj:`NotImplemented` when the structure does not apply.
        subspace : The name of the boolean attribute that :meth:`weaken`
            restricts the morphisms of the law to, if any.
        broken : Whether the law is declared broken by :meth:`failing`.
    """

    _: KW_ONLY
    subspace: str | None = None
    broken: bool = False

    concludes = False

    __hash__ = Declaration.__hash__

    def __get__(self, instance, owner: type) -> Self:
        declaring = next((
            base for base in owner.__mro__
            if base.__dict__.get(self.name or "") is self), None)
        return self.bind(owner, owner=declaring)

    def modulo(self, up_to) -> Self:
        """
        The same law with its equation compared up to a function, so that a
        category weakens an inherited axiom in one statement, e.g. a diagram
        compares the interchange law up to its normal form:
        ``Diagram.bifunctoriality = MonoidalCategory.bifunctoriality.modulo(
        Diagram.normal_form)``.
        """
        @wraps(self.function)
        def equation(*args, **kwargs):
            return self.function(*args, **kwargs).modulo(up_to)
        return replace(self, function=equation)

    def failing(self, reason: str) -> Self:
        """
        The same law declared broken: calling it raises an
        :class:`AxiomFailure` with the reason as message and the equation
        evaluated on the arguments, e.g. ``braid_naturality =
        BraidedCategory.braid_naturality.failing("A free braid is a box.")``.
        """
        @wraps(self.function)
        def equation(*args, **kwargs):
            raise AxiomFailure(reason, self.function(*args, **kwargs))
        equation.__doc__ = reason
        return replace(self, function=equation, broken=True)

    def inapplicable(self, reason: str) -> Self:
        """
        The same law declared not to apply to the category: it takes no
        argument and returns :obj:`NotImplemented`, with the reason as its
        documentation, e.g. ``trace_vanishing =
        TracedCategory.trace_vanishing.inapplicable("No trace.")``.
        """
        def law(cls):
            # pylint: disable=unused-argument  # a law that does not apply
            return NotImplemented
        law.__doc__ = reason
        return replace(self, function=law, subspace=None, broken=False)

    def weaken(self, subspace: str) -> Self:
        """
        The same law quantified over the subspace of the morphisms whose
        attribute of that name holds, both its premises and the terms of
        its equation, e.g. ``bifunctoriality.weaken(
        "is_boundary_connected")`` on a diagram category compares the
        interchange law on the diagrams its normal form is defined for: a
        state and an effect are each boundary-connected while composing
        into a closed component, on which it is not. Assigned to its own
        attribute beside a ``.failing`` declaration, it shows the matrix
        one expected failure and one green cell instead of one blanket
        expected failure.
        """
        return replace(self, subspace=subspace)

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """
        The parameters whose arguments the property matrix generates: all
        but the first, which is the category.
        """
        return tuple(
            inspect.signature(self.function).parameters.values())[1:]

    def equations(self, evaluate: Callable) -> "st.SearchStrategy":
        """ The law evaluated by a function of its arguments, sampled by
        :meth:`Testable.sample`, in the subspace it was weakened to. """
        equations = self.bound.sample(self).map(
            lambda arguments: evaluate(**arguments))
        if self.subspace is None:
            return equations
        holds = attrgetter(self.subspace)
        return equations.filter(
            lambda equation: all(map(holds, equation.terms)))

    def strategy(self) -> "st.SearchStrategy[Equation]":
        """
        Generate the equations the bound axiom states: a law declared
        broken raises its :class:`AxiomFailure` from the sample, as it does
        from a call.

        >>> from hypothesis import find
        >>> from discopy.cat import Arrow
        >>> equation = find(Arrow.unitality.strategy(), lambda _: True)
        >>> assert equation and len(equation.terms) == 3
        """
        return self.equations(self)

    def falsify(self, **params) -> Equation:
        """
        Search for a shrunk counterexample to the bound axiom, a false
        equation of the law, raising :class:`hypothesis.errors.NoSuchExample`
        when none is found. Keyword arguments are passed to
        :func:`hypothesis.find`.

        >>> from hypothesis import settings
        >>> from discopy.cat import Arrow
        >>> Arrow.associativity.falsify(
        ...     settings=settings(max_examples=10))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
         ...
        hypothesis.errors.NoSuchExample: No examples found of condition ...
        """
        from hypothesis import find

        def verdict(**arguments):
            try:
                return self(**arguments)
            except AxiomFailure as failure:
                return failure.equation

        return find(
            self.equations(verdict),
            lambda equation: equation is not NotImplemented
            and not equation, **params)

    def canonical(self) -> Equation:
        """
        The law as a schema: its equation on the canonical arguments of its
        sequent, a box per premise between objects named after the
        variables, the equation a law declared broken raises included.

        >>> from discopy.cat import Arrow
        >>> print(Arrow.associativity.canonical())
        Equation(f >> g >> h, f >> g >> h)
        """
        try:
            return self(**super().canonical())
        except AxiomFailure as failure:
            return failure.equation

    def draw(self, **params):
        """
        Draw the :meth:`canonical` equation of the law, the parameters those
        of :meth:`discopy.monoidal.Equation.draw`. A law the category breaks
        on its canonical instance is drawn all the same, as the inequation it
        is there, see :meth:`Equation.checked`.

        >>> from discopy import braided
        >>> braided.Diagram.braid_naturality.draw(
        ...     doctest="docs/_static/braided/braid-naturality-failing.svg")

        .. image:: /_static/braided/braid-naturality-failing.svg
            :align: center
        """
        equation = self.canonical()
        if equation is NotImplemented:
            raise TypeError(f"{self} does not apply, so has nothing to draw.")
        return equation.checked().draw(**params)

    def arguments(self, /, *args: P.args, **kwargs: P.kwargs) -> dict:
        """ Bind the arguments to the :attr:`parameters` of the axiom. """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        bound = inspect.Signature(self.parameters).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def __call__(self, /, *args: P.args, **kwargs: P.kwargs):
        return self.function(self.category, **self.arguments(*args, **kwargs))


def axiom[**P, T](function: Callable[P, T]) -> Axiom[P, T]:
    """
    Decorate an equation as a categorical axiom: a classmethod of its
    category, implicitly, whose remaining parameters are generated.
    """
    return Axiom(function)


class Serialisable(Testable):
    """
    The serialisation interface of DisCoPy, one hook driving all three
    mechanisms: the class attribute ``serialised_attrs`` names attributes that
    are also keyword arguments of ``__init__``, from which follow

    - a generic pair of inverse methods :meth:`to_tree` and
      :meth:`from_tree`, the JSON serialisation behind
      :func:`discopy.utils.dumps` and :func:`discopy.utils.loads`,
    - a generic :meth:`__repr__` such that ``eval(repr(x)) == x``,
    - :meth:`__setstate__` for the pickle protocol; the class
      parameters of :class:`discopy.utils.NamedGeneric` are pickled by
      their own machinery.

    A subclass with a different constructor declares its keys once
    instead of reimplementing each method.

    Each mechanism comes with the law that it is a roundtrip, i.e. that
    a term reads back from what it was written to:
    :meth:`repr_transparency`
    for its representation, :meth:`pickling` and :meth:`copying` for the
    pickle protocol and :meth:`serialisation` for its tree. They are
    axioms like any other, so a class that also implements
    :meth:`discopy.axioms.Testable.strategy` has them checked against
    generated terms, and one that violates a law declares it
    ``.failing`` rather than leaving it untested.

    Example
    -------
    >>> from discopy.cat import Box
    >>> assert Box.serialised_attrs\\
    ...     == ('name', 'dom', 'cod', 'is_dagger', 'data')
    >>> f = Box('f', 'x', 'y', data=42)
    >>> assert Box.from_tree(f.to_tree()) == f
    """
    serialised_attrs: tuple[str, ...] = ()

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace the representation of a term reads back in: the
        public names of the package, as ``from discopy import *`` binds
        them, so that a representation qualified by module such as
        ``cat.Box('f', cat.Ob('x'), cat.Ob('y'))`` evaluates, and then
        those of the module the class is defined in, so that one
        printing bare names such as ``Tensor[int]([0], dom=Dim(1),
        cod=Dim(1))`` evaluates too. The module comes second because a
        term prints the names its own module binds: ``Dim`` in
        ``discopy.tensor`` is the one a tensor is built from.

        The import is local because the package imports this module.
        """
        import discopy

        public = lambda namespace: {
            name: value for name, value in namespace.items()
            if not name.startswith("_")}
        module = sys.modules[cls.__module__]
        return dict(public(vars(discopy)), **public(vars(module)))

    def is_default(self, key: str) -> bool:
        """
        Whether the value of an attribute equals its class default,
        in which case :meth:`to_tree` and :meth:`__repr__` drop it.

        Parameters:
            key : The name of the attribute.
        """
        if not hasattr(type(self), key):
            return False
        value, default = getattr(self, key), getattr(type(self), key)
        return value is default or (
            type(value) is type(default) and value == default)

    def __repr__(self):
        """
        The transparent representation of a DisCoPy object: an attribute
        without a class default is positional, one that differs from its
        default is a keyword argument and one equal to it is dropped.

        Example
        -------
        >>> import discopy
        >>> from discopy.cat import Box
        >>> f = Box('f', 'x', 'y', data=42)
        >>> f
        cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42)
        >>> assert eval(repr(f), vars(discopy)) == f
        """
        return factory_name(type(self)) + "(" + ", ".join(
            f"{key}={repr(getattr(self, key))}" if hasattr(type(self), key)
            else repr(getattr(self, key))
            for key in self.serialised_attrs if not self.is_default(key)) + ")"

    def __setstate__(self, state):
        """
        Restore a pickled state.

        Parameters:
            state : The pickled state of the object.
        """
        self.__dict__.update(state)

    def to_tree(self) -> dict:
        """
        Serialise a DisCoPy object, see :func:`dumps`.

        The tree records the :func:`factory_name` and then each of the
        ``serialised_attrs``, dropping a key when its value equals the
        class attribute of the same name, e.g. a box that is not a
        dagger. An attribute with a ``to_tree`` method is serialised, a
        non-empty list or tuple of such attributes becomes the list of
        their trees, raw JSON data passes through unchanged.

        Example
        -------
        >>> from pprint import PrettyPrinter
        >>> pprint = PrettyPrinter(indent=4, width=70, sort_dicts=False).pprint
        >>> from discopy.cat import Box
        >>> f = Box('f', 'x', 'y', data=42)
        >>> pprint((f >> f[::-1]).to_tree())
        {   'factory': 'cat.Arrow',
            'inside': [   {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'x'},
                              'cod': {'factory': 'cat.Ob', 'name': 'y'},
                              'data': 42},
                          {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'y'},
                              'cod': {'factory': 'cat.Ob', 'name': 'x'},
                              'is_dagger': True,
                              'data': 42}],
            'dom': {'factory': 'cat.Ob', 'name': 'x'},
            'cod': {'factory': 'cat.Ob', 'name': 'x'}}
        """
        tree = {'factory': factory_name(type(self))}
        for key in self.serialised_attrs:
            if self.is_default(key):
                continue
            value = getattr(self, key)
            if hasattr(value, 'to_tree'):
                value = value.to_tree()
            elif isinstance(value, (list, tuple)) and value and all(
                    hasattr(v, 'to_tree') for v in value):
                value = [v.to_tree() for v in value]
            tree[key] = value
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Serialisable:
        """
        Decode a serialised DisCoPy object, see :func:`loads`.

        A key missing from the tree falls back to the default value of
        the corresponding keyword argument of ``__init__``. A value with
        a ``'factory'`` key decodes recursively, a non-empty list of
        such values to the tuple of decoded objects, raw JSON data
        passes through unchanged.

        Parameters:
            tree : DisCoPy serialisation.

        Example
        -------
        >>> from discopy.cat import Ob
        >>> assert Ob.from_tree({'factory': 'cat.Ob', 'name': 'x'}) == Ob('x')
        """
        from discopy.utils import from_tree

        kwargs = {}
        for key in cls.serialised_attrs:
            if key not in tree:
                continue
            value = tree[key]
            if isinstance(value, dict) and 'factory' in value:
                value = from_tree(value)
            elif isinstance(value, list) and value and all(
                    isinstance(v, dict) and 'factory' in v for v in value):
                value = tuple(map(from_tree, value))
            kwargs[key] = value
        return cls(**kwargs)

    @axiom
    def repr_transparency(cls, term: Self) -> Equation:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.

        Its ``str`` reads back too, which STYLE.md asks of every term
        under "the obvious variable naming convention", but the
        environment that convention needs is more than a namespace, so
        ``str_transparency`` is left to
        `#764 <https://github.com/discopy/discopy/issues/764>`_.
        """
        return Equation(eval(repr(term), cls.environment()), term)

    @axiom
    def pickling(cls, term: Self) -> Equation:
        """
        A term loads back from its pickle, of the same class: the equation
        is between the pairs of a class and a term, since a subscript of a
        :class:`discopy.utils.NamedGeneric` is part of what a pickle keeps.
        """
        loaded = pickle.loads(pickle.dumps(term))
        return Equation((type(loaded), loaded), (type(term), term))

    @axiom
    def copying(cls, term: Self) -> Equation:
        """
        A term is equal to its deep copy, of the same class. Copying goes
        through the same protocol as :meth:`pickling` without the bytes,
        so a class whose reduction drops what its state needs — the
        parameters of a :class:`discopy.utils.NamedGeneric`, say — breaks
        one law with the other.
        """
        copied = deepcopy(term)
        return Equation((type(copied), copied), (type(term), term))

    @axiom
    def serialisation(cls, term: Self) -> Equation:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        from discopy.utils import dumps, from_tree, loads

        return Equation(from_tree(term.to_tree()), loads(dumps(term)), term)
