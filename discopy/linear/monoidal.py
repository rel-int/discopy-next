# -*- coding: utf-8 -*-

"""
The free linear category, i.e. a monoidal category with a bang: a comonad
``!`` whose objects are comonoids, central and lax monoidal, see
:class:`discopy.abc.LinearCategory`. The comonad, its :class:`Bang`,
:class:`Dereliction` and :class:`Digging`, are those of
:mod:`discopy.linear.cat`; this module adds what uses the tensor, the
:class:`Contraction` and :class:`Weakening` of the comonoid on a banged
object, the :class:`Laxator` and the :class:`Exchange`, which makes a banged
object central and needs no symmetry.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    BangOb
    Diagram
    Box
    Sum
    Bang
    Structural
    Dereliction
    Digging
    Contraction
    Weakening
    Exchange
    Laxator
    Functor
    TermBase
    Variable
    Constant
    Application
    Kappa

Example
-------
The translation of the non-linear term ``lambda x: (f(x), f(x))`` banged its
argument: it copies ``!x`` by contraction and reads each copy once by
dereliction, after the bang of ``f``.

>>> x, y = Ty('x'), Ty('y')
>>> f = Box('f', x, y)
>>> twice = f.bang() >> Contraction(y) >> Dereliction(y) @ Dereliction(y)
>>> assert twice.dom == x.bang() and twice.cod == y @ y
>>> twice.draw(doctest='docs/_static/linear/twice.svg')

.. image:: /_static/linear/twice.svg
    :align: center

A :class:`Functor` into a category without a bang, e.g.
:class:`discopy.tensor.Diagram`, sends banged wires and boxes through its
``ob_map`` and ``ar_map``, e.g. ``!x`` to a Fock space and ``f.bang()`` to
the second quantisation of ``f``, see the notebook
:doc:`/notebooks/optyx/linear_logic`.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Self

from discopy import monoidal, messages
from discopy.abc import DaggerCategory, LinearCategory, MonoidalCategory
from discopy.cat import factory
from discopy.drawing import Drawing
from discopy.linear import cat
from discopy.utils import (
    AxiomError, assert_isinstance, get_origin)


@factory
class Ty(monoidal.Ty):
    """
    A type with a bang, a monoidal type whose wires may be
    :class:`BangOb`.
    """
    bang_factory: ClassVar[type[BangOb]]
    term_factory: ClassVar[type["TermBase"]]

    def __call__(self, function: Callable, offset: int | None = None):
        """
        The kappa abstraction of a Python function, a :class:`Kappa` of the
        terms of its :attr:`term_factory`: a type with a bang binds its
        variables in a linear category even when the body has no box, e.g.
        weakens a banged variable it does not use.

        Parameters:
            function : The body, as a function of the variables.
            offset : Passed to :class:`Kappa`.

        Example
        -------
        >>> x = Ty('x')
        >>> term = (x.bang() @ x)(lambda a, b: b)
        >>> assert isinstance(term, Kappa)
        >>> assert term.eval() == Weakening(x) @ x
        >>> assert eval(str(term)) == term
        """
        return self.term_factory.kappa_factory.from_callable(
            self, offset)(function)

    def bang(self) -> Self:
        """
        The bang of a type, a single wire :class:`BangOb` holding it: the
        bang is lax monoidal, ``!(x @ y)`` is one wire and not ``!x @ !y``.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert x.bang() == Ty(BangOb(x)) != Ty(BangOb(x @ y))
        >>> assert (x @ y).bang() != x.bang() @ y.bang()
        >>> assert str(x.bang()) == "x.bang()"
        >>> assert eval(str((x @ y).bang().bang())) == (x @ y).bang().bang()
        """
        return type(self)(self.bang_factory(self))


class BangOb(cat.BangOb, monoidal.Wire):
    """
    The bang ``!x`` of a type ``x``, a wire of its own holding it, the way an
    :class:`Exp` holds its base and exponent.

    Parameters:
        arg : The type to bang.

    Example
    -------
    >>> from discopy import linear
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Ty(BangOb(x @ y)) == (x @ y).bang()
    >>> assert eval(repr(BangOb(x))) == BangOb(x)
    >>> assert BangOb(x @ y).drawing_name == "!(x @ y)"
    """
    ob = Ty

    def __str__(self):
        arg = str(self.arg) if len(self.arg) == 1 else f"({self.arg})"
        return arg + ".bang()"

    @property
    def drawing_name(self) -> str:
        """ The label of a banged wire, e.g. ``!x`` or ``!(x @ y)``. """
        arg = self.arg.to_drawing()
        return "!" + (str(arg) if len(arg) == 1 else f"({arg})")

    def atoms(self) -> list:
        """ The atoms of a bang: those of the type it holds. """
        return self.arg.atoms()

    def dagger(self) -> BangOb:
        """ A banged wire is its own dagger, as it has no colour. """
        return self


Ty.bang_factory = BangOb


@factory
class Diagram(monoidal.Diagram, cat.Arrow, LinearCategory):
    """
    A diagram in the free linear category: a monoidal diagram with
    :class:`Bang` bubbles and the :class:`Dereliction`, :class:`Digging`,
    :class:`Contraction`, :class:`Weakening`, :class:`Exchange` and
    :class:`Laxator` boxes, see :class:`discopy.abc.LinearCategory`.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    sum_factory: ClassVar[type["Sum"]]
    functor_factory: ClassVar[type["Functor"]]
    bang_factory: ClassVar[type["Bang"]]
    dereliction_factory: ClassVar[type["Dereliction"]]
    digging_factory: ClassVar[type["Digging"]]
    contraction_factory: ClassVar[type["Contraction"]]
    weakening_factory: ClassVar[type["Weakening"]]
    exchange_factory: ClassVar[type["Exchange"]]
    laxator_factory: ClassVar[type["Laxator"]]

    ob = Ty
    dagger_involution = DaggerCategory.dagger_involution.failing(
        messages.STRUCTURAL_HAS_NO_DAGGER)
    dagger_contravariance = DaggerCategory.dagger_contravariance.failing(
        messages.STRUCTURAL_HAS_NO_DAGGER)

    @classmethod
    def contraction(cls, x: Ty) -> Contraction:
        """ Wrapper around :class:`Contraction`. """
        return cls.contraction_factory(x)

    @classmethod
    def weakening(cls, x: Ty) -> Weakening:
        """ Wrapper around :class:`Weakening`. """
        return cls.weakening_factory(x)

    @classmethod
    def exchange(cls, x: Ty, y: Ty, is_dagger: bool = False) -> Exchange:
        """ Wrapper around :class:`Exchange`. """
        return cls.exchange_factory(x, y, is_dagger)

    @classmethod
    def laxator(cls, x: Ty | None = None, y: Ty | None = None) -> Laxator:
        """ Wrapper around :class:`Laxator`. """
        return cls.laxator_factory(x, y)

    @classmethod
    def copy(cls, x: Ty, n: int = 2) -> Diagram:
        """
        The ``n`` copies of a banged wire ``!y``, by :class:`Contraction`, so
        that a term language sharing its variables with ``cod.copy`` and
        ``cod.discard``, e.g. :class:`discopy.closed.markov.TermBase`, copies
        only what linear logic lets it copy.

        Parameters:
            x : The banged wire to copy.
            n : The number of copies.

        Example
        -------
        >>> y = Ty('y')
        >>> assert Diagram.copy(y.bang(), 3) == Contraction(y)\\
        ...     >> y.bang() @ Contraction(y)
        >>> assert Diagram.copy(y.bang(), 0) == Weakening(y)
        >>> Diagram.copy(y.bang(), -1)
        Traceback (most recent call last):
        ...
        ValueError: Expected a number of copies, got -1.
        >>> Diagram.copy(y)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: y is not banged: ...
        """
        if len(x) != 1 or not isinstance(x.inside[0], BangOb):
            raise AxiomError(messages.NOT_BANGED.format(x))
        if n < 0:
            raise ValueError(messages.NEGATIVE_COPIES.format(n))
        arg = x.inside[0].arg
        if not n:
            return cls.weakening(arg)
        result = cls.id(x)
        for i in range(n - 1):
            result = result >> x ** i @ cls.contraction(arg)
        return result

    @classmethod
    def discard(cls, x: Ty) -> Diagram:
        """
        The discard of a tensor of banged wires, by :class:`Weakening`.

        Parameters:
            x : The banged wires to discard.

        Example
        -------
        >>> y, z = Ty('y'), Ty('z')
        >>> assert Diagram.discard(y.bang() @ z.bang())\\
        ...     == Weakening(y) @ Weakening(z)
        """
        return cls.id(x[:0]).tensor(*(
            cls.copy(x[i:i + 1], 0) for i in range(len(x))))

    @classmethod
    def wiring(cls, wiring: list[int], dom: Ty) -> Diagram:
        """
        The wiring of a type to the wires at the given indices, where only a
        banged wire is copied, discarded or moved: the copies of each banged
        wire by :meth:`copy`, then the exchanges that bring each copy where
        it is read. A wire that is not banged is read once and in order.

        Parameters:
            wiring : The index of the input each output reads.
            dom : The type of the inputs.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.wiring([0, 1, 0], x.bang() @ y)\\
        ...     == Contraction(x) @ y >> x.bang() @ Exchange(x, y)
        >>> Diagram.wiring([1, 0], x @ y)
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: x @ y is not banged: ...
        """
        if any(not 0 <= i < len(dom) for i in wiring):
            raise ValueError(messages.WIRING_OUT_OF_RANGE.format(dom, wiring))
        wires = [dom[i:i + 1] for i in range(len(dom))]
        banged = [isinstance(wire.inside[0], BangOb) for wire in wires]
        counts = [wiring.count(i) for i in range(len(dom))]
        if any(n != 1 and not b for n, b in zip(counts, banged)):
            raise AxiomError(messages.NOT_BANGED.format(dom))
        result = cls.id(dom[:0]).tensor(*(
            cls.copy(wire, n) if b else cls.id(wire)
            for wire, n, b in zip(wires, counts, banged)))
        targets = {i: [j for j, k in enumerate(wiring) if k == i]
                   for i in range(len(dom))}
        order = [j for i in range(len(dom)) for j in targets[i]]
        types = [wires[wiring[j]] for j in order]
        for _ in order:
            for i in range(len(order) - 1):
                if order[i] < order[i + 1]:
                    continue
                left, right = types[i], types[i + 1]
                result = result >> dom[:0].tensor(*types[:i])\
                    @ cls.crossing(left, right)\
                    @ dom[:0].tensor(*types[i + 2:])
                order[i], order[i + 1] = order[i + 1], order[i]
                types[i], types[i + 1] = right, left
        return result

    @classmethod
    def crossing(cls, left: Ty, right: Ty) -> Diagram:
        """
        The crossing of two adjacent wires that :meth:`wiring` moves past
        each other, an :meth:`exchange` of the banged one: two wires that are
        not banged do not cross.

        Parameters:
            left : The wire on the left.
            right : The wire on the right.
        """
        if isinstance(left.inside[0], BangOb):
            return cls.exchange(left.inside[0].arg, right)
        if isinstance(right.inside[0], BangOb):
            return cls.exchange(right.inside[0].arg, left, True)
        raise AxiomError(messages.NOT_BANGED.format(left @ right))

    def __call__(self, *arguments: TermBase) -> TermBase:
        """
        A box called on terms is its :class:`Constant` applied to them, so
        that a term is written with the boxes it calls, e.g. ``f.bang()(x)``.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> print(x.bang()(lambda a: Dereliction(y)(f.bang()(a))))
        x.bang()(lambda a: Dereliction(y)((f).bang()(a)))
        """
        return self.term_factory.constant_factory(self)(*arguments)


class Box(monoidal.Box, cat.Box, Diagram):
    """
    A box in the free linear category, i.e. a monoidal box of its diagrams.
    """


class Sum(monoidal.Sum, Box):
    """
    A sum in the free linear category, i.e. a monoidal sum and its box.
    """


class Bang(cat.Bang, Box):
    """
    The bang ``!f : !x -> !y`` of a diagram ``f : x -> y``, a box with ``f``
    as its arg, drawn with an exclamation mark.

    Parameters:
        arg : The diagram to bang.

    Example
    -------
    >>> from discopy import linear
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Bang(f).dom == x.bang() and Bang(f).cod == y.bang()
    >>> assert eval(repr(Bang(f))) == Bang(f) and eval(str(Bang(f))) == Bang(f)

    The promotion of !L*, from ``!x -> y`` to ``!x -> !y``, is the digging
    followed by the bang:

    >>> g = Box('g', x.bang(), y)
    >>> promotion = Digging(x) >> g.bang()
    >>> assert promotion.dom == x.bang() and promotion.cod == y.bang()
    """
    # pylint: disable-next=super-init-not-called  # a monoidal box is drawn
    def __init__(self, arg: Diagram):
        dom, cod = arg.dom.bang(), arg.cod.bang()
        Box.__init__(self, f"({arg}).bang()", dom, cod, args=(arg, ),
                     drawing_name="!")


class Structural(cat.Structural, Box):
    """
    A structural box of a linear category, a structural box of its comonad
    and a box of its diagrams.
    """


class Dereliction(cat.Dereliction, Structural):
    """
    The dereliction ``!x -> x`` of a type, see
    :class:`discopy.linear.cat.Dereliction`.

    Parameters:
        x : The type to derelict.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Dereliction(x).dom == x.bang() and Dereliction(x).cod == x
    """
    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the box at an atomic type.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        return cls.atomic_strategy(types).map(cls)


class Digging(cat.Digging, Structural):
    """
    The digging ``!x -> !!x`` of a type, see
    :class:`discopy.linear.cat.Digging`.

    Parameters:
        x : The type to dig.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Digging(x).cod == x.bang().bang()
    """
    strategy = classmethod(Dereliction.strategy.__func__)


class Contraction(Structural):
    """
    The contraction ``!x -> !x @ !x``, the rule of !L* deriving
    ``Γ, !x, Δ ⊢ y`` from ``Γ, !x, !x, Δ ⊢ y``.

    Parameters:
        x : The type to contract.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Contraction(x).cod == x.bang() @ x.bang()
    """
    serialised_attrs = ("x", )
    drawing_name = "Δ"
    strategy = classmethod(Dereliction.strategy.__func__)

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Contraction({x})", x.bang(), x.bang() @ x.bang())


class Weakening(Structural):
    """
    The weakening ``!x -> Ty()``, the counit of the contraction. !L* is
    relevant: it has no weakening, which an example of it never uses.

    Parameters:
        x : The type to weaken.

    Example
    -------
    >>> x = Ty('x')
    >>> assert Weakening(x).dom == x.bang() and not Weakening(x).cod
    """
    serialised_attrs = ("x", )
    drawing_name = "e"
    strategy = classmethod(Dereliction.strategy.__func__)

    def __init__(self, x: Ty):
        self.x = x
        super().__init__(f"Weakening({x})", x.bang(), x[:0])


class Exchange(Structural):
    """
    The exchange ``!x @ y -> y @ !x`` of a banged type with any type, and its
    inverse ``y @ !x -> !x @ y`` when ``is_dagger``, the permutation rule of
    !L*: it makes ``!x`` central rather than the category symmetric.

    Parameters:
        x : The type that is banged.
        y : The type it is exchanged with.
        is_dagger : Whether to take the inverse.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Exchange(x, y).dom == x.bang() @ y
    >>> assert Exchange(x, y).dagger() == Exchange(x, y, True)
    >>> assert Exchange(x, y, True).dom == y @ x.bang()
    >>> assert eval(str(Exchange(x, y)[::-1])) == Exchange(x, y, True)
    """
    serialised_attrs = ("x", "y", "is_dagger")
    drawing_name = "σ"

    def __init__(self, x: Ty, y: Ty, is_dagger: bool = False):
        self.x, self.y = x, y
        dom, cod = x.bang() @ y, y @ x.bang()
        super().__init__(f"Exchange({x}, {y})", *(
            (cod, dom) if is_dagger else (dom, cod)), is_dagger=is_dagger)

    def dagger(self) -> Exchange:
        return type(self)(self.x, self.y, not self.is_dagger)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the exchange of two atomic types, or its inverse.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.builds(cls, atoms, atoms, st.booleans())


class Laxator(Structural):
    """
    The laxator ``!x @ !y -> !(x @ y)`` of the bang, and its unit
    ``Ty() -> !Ty()`` when given no type.

    Parameters:
        x : The type on the left.
        y : The type on the right.

    Example
    -------
    >>> from discopy import linear
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Laxator(x, y).dom == x.bang() @ y.bang()
    >>> assert Laxator(x, y).cod == (x @ y).bang()
    >>> assert not Laxator().dom and Laxator().cod == Ty().bang()
    >>> assert eval(repr(Laxator())) == Laxator()
    """
    serialised_attrs = ("x", "y")
    drawing_name = "m"

    def __init__(self, x: Ty | None = None, y: Ty | None = None):
        if (x is None) != (y is None):
            raise ValueError(messages.LAXATOR_ARITY.format((x, y)))
        self.x, self.y = x, y
        if x is None or y is None:
            name, dom = "Laxator()", self.ob()
            cod = Ty.bang(dom)
        else:
            name, dom = f"Laxator({x}, {y})", x.bang() @ y.bang()
            cod = (x @ y).bang()
        super().__init__(name, dom, cod)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the laxator of two atomic types, or its unit.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.one_of(st.just(cls()), st.builds(cls, atoms, atoms))


Id = Diagram.id
Diagram.box_factory = Box
Diagram.sum_factory = Sum
Diagram.bang_factory = Bang
Diagram.dereliction_factory = Dereliction
Diagram.digging_factory = Digging
Diagram.contraction_factory = Contraction
Diagram.weakening_factory = Weakening
Diagram.exchange_factory = Exchange
Diagram.laxator_factory = Laxator
for law in sorted(
        set(LinearCategory.axioms) - set(MonoidalCategory.axioms)):
    setattr(Diagram, law, getattr(LinearCategory, law).failing(
        messages.FREE_BANG))


class Functor(monoidal.Functor):
    """
    A functor from the free linear category: into a linear category it
    sends a :class:`BangOb`, a :class:`Bang` and the structural boxes
    through the bang and the structure of its codomain, and into any other
    category, e.g. :class:`discopy.tensor.Diagram`, through its ``ob_map``
    and ``ar_map``.

    Example
    -------
    >>> from discopy import tensor
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> F = Functor({x: y, y: x}, {f: f[::-1]})
    >>> assert F(Contraction(x) >> f.bang() @ Weakening(x)) == (
    ...     Contraction(y) >> f[::-1].bang() @ Weakening(y))
    >>> e = tensor.Box('e', tensor.Dim(3), tensor.Dim(2))
    >>> G = Functor(ob_map={x: tensor.Dim(2), x.bang(): tensor.Dim(3)},
    ...             ar_map={Dereliction(x): e}, cod=tensor.Diagram)
    >>> assert G(Dereliction(x)) == e
    """
    dom = cod = Diagram

    @property
    def has_bang(self) -> bool:
        """
        Whether the codomain is a linear category, which a :class:`BangOb`,
        a :class:`Bang` and the structural boxes are sent through.
        """
        return issubclass(get_origin(self.cod), LinearCategory)

    def map_bang(self, other):
        """
        The image of a :class:`BangOb`, a :class:`Bang` or a structural box
        in a codomain with a bang, see :attr:`has_bang`.

        Parameters:
            other : The wire or box to map.
        """
        if isinstance(other, BangOb):
            return self.cod.banged(self(other.arg))
        if isinstance(other, Bang):
            return self(other.arg).bang()
        if isinstance(other, Laxator):
            return self.cod.laxator() if other.x is None\
                else self.cod.laxator(self(other.x), self(other.y))
        if isinstance(other, Exchange):
            return self.cod.exchange(
                self(other.x), self(other.y), other.is_dagger)
        method = getattr(self.cod, type(other).__name__.lower())
        return method(self(other.x))

    def __call__(self, other):
        if isinstance(other, (BangOb, Bang, Structural))\
                and self.cod is not Drawing:
            if self.has_bang:
                return self.map_bang(other)
            if isinstance(other, Bang):
                return self.ar_map[other]
        return super().__call__(other)


Diagram.functor_factory = Functor


class TermBase(monoidal.TermBase):
    """
    A term of a linear category, i.e. of the planar kappa calculus with a
    bang: a :class:`discopy.monoidal.TermBase` where a variable of a banged
    type ``!x`` occurs any number of times, every other variable once and in
    order. An :class:`Application` copies the banged variables its function
    and its argument share by :class:`Contraction` and brings each copy
    where it is read by :class:`Exchange`, see :meth:`Diagram.wiring`.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y), Box('g', y.bang() @ y.bang(), x)
    >>> term = x.bang()(lambda a: g(f.bang()(a), f.bang()(a)))
    >>> assert term.eval() == Contraction(x) >> x.bang() @ f.bang()\\
    ...     >> f.bang() @ y.bang() >> g
    >>> h = Box('h', x @ x, y)
    >>> x(lambda a: h(a, a))
    Traceback (most recent call last):
    ...
    ValueError: Expected banged variables to share, got a.
    """
    @staticmethod
    def wire(functor, wiring: list[int], dom: Ty):
        """
        The image under a functor of the wiring from a type to its wires at
        the given indices, see :meth:`Diagram.wiring`.
        """
        return functor(Diagram.wiring(wiring, dom))


class Variable(TermBase, monoidal.Variable):
    """
    A variable with a name and a type of one wire, an identity diagram.

    Parameters:
        name : The name of the variable.
        cod : The type of the variable.
    """


class Constant(TermBase, monoidal.Constant):
    """
    A box as a term, with its domain as stack.

    Parameters:
        box : The box.
    """


class Application(TermBase, monoidal.Application):
    """
    The application of a term to an argument with no stack, which shares
    their banged variables: their context is that of the function with the
    variables of the argument where its stack was, each one where it first
    occurs.

    Parameters:
        func : The term that is applied.
        arg : The argument, a term with no stack.
    """
    @staticmethod
    def share(*contexts):
        """ The distinct free variables of terms side by side. """
        variables = [x for context in contexts for x in context]
        shared = [x for x in dict.fromkeys(variables)
                  if variables.count(x) > 1]
        if any(not isinstance(x.cod.inside[0], BangOb) for x in shared):
            raise ValueError(messages.NOT_SHARED.format(
                ", ".join(x.name for x in shared)))
        return list(dict.fromkeys(variables))

    @staticmethod
    def contractible(func, arg):
        """
        Whether a function applied to an argument is a redex to contract:
        one that uses its variable exactly once, which is not copied.
        """
        return monoidal.Application.contractible(func, arg)\
            and func.body.count(func.head) == 1

    def eval(self, functor=None):
        functor = self.identity(functor)
        left, _, right = self.func.context
        stack = list(range(len(self.stack)))
        before, _, after = self.context
        source = before + stack + after
        wiring = [source.index(wire)
                  for wire in left + self.arg.freevars + stack + right]
        return self.wire(functor, wiring, self.dom) >> super().eval(functor)


class Kappa(TermBase, monoidal.Kappa):
    """
    The abstraction of a block of ``variables`` of the ``body``, as in a
    monoidal category, where a banged variable the body does not use is
    discarded by :class:`Weakening`: the ones it uses are a block in order,
    the others stand among them in the stack.

    Parameters:
        variables : The variables to bind, in the order of the stack.
        body : The term in which they are bound.
        offset : The number of free variables left of the stack, needed
            when nothing says where it is: no stack in the body, which uses
            none of the variables.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> term = (x.bang() @ x)(lambda a, b: f(b))
    >>> assert term.stack == x.bang() @ x
    >>> assert term.eval() == Weakening(x) @ x >> f
    >>> assert x.bang()(lambda a: Constant(f), offset=0).eval()\\
    ...     == Weakening(x) @ f
    >>> (x @ x)(lambda a, b: f(b))
    Traceback (most recent call last):
    ...
    ValueError: Expected a block of variables next to the stack of f(b), \
got a, b.
    """
    above = None

    @staticmethod
    def used(variables, body) -> tuple:
        """ The variables the body uses, in the order they are bound. """
        return tuple(x for x in variables if x in body.freevars)

    @staticmethod
    def binds(variables, body):
        """
        Whether a body has the variables to bind: distinct ones, each free
        in it or banged, so that it is discarded when the body does not use
        it.
        """
        return len(set(variables)) == len(variables) and all(
            x in body.freevars or isinstance(x.cod.inside[0], BangOb)
            for x in variables)

    @classmethod
    def place(cls, variables, body):
        """
        The number of free variables left of the stack: where the first
        variable the body uses stands in its context, or else where its
        stack does, ``None`` when it has none.
        """
        used = cls.used(variables, body)
        return body.freevars.index(used[0]) if used else body.offset

    def __init__(self, variables: tuple, body: monoidal.TermBase,
                 offset: int | None = None):
        # pylint: disable=super-init-not-called  # unused variables too
        assert_isinstance(body, monoidal.TermBase)
        variables = tuple(variables)
        used = self.used(variables, body)
        start = self.start(variables, body, offset)
        stop = start + len(used)
        if tuple(body.freevars[start:stop]) != used\
                or body.offset not in (None, start, stop):
            raise ValueError(self.unbound.format(
                body, ", ".join(x.name for x in variables)))
        self.variables, self.body, self.offset = variables, body, start
        self.freevars = body.freevars[:start] + body.freevars[stop:]
        self.cod, bound = body.cod, body.types(variables)
        self.stack = bound @ body.stack if self.front else body.stack @ bound

    @property
    def front(self) -> bool:
        """
        Whether the variables join the stack at its front: they do unless
        the ones the body uses stand right of its stack.
        """
        right = self.body.context[2] if self.body.stack else []
        return not any(
            x in right for x in self.used(self.variables, self.body)[:1])

    def eval(self, functor=None):
        functor, body = self.identity(functor), self.body
        stack = list(range(len(body.stack)))
        bound = list(self.variables)
        left, _, right = body.context
        source = self.freevars[:self.offset] + (
            bound + stack if self.front else stack + bound
        ) + self.freevars[self.offset:]
        wiring = [source.index(wire) for wire in left + stack + right]
        return self.wire(functor, wiring, self.dom) >> body.eval(functor)


TermBase.functor = Functor.id(Diagram)
TermBase.variable_factory, TermBase.constant_factory = Variable, Constant
TermBase.application_factory, TermBase.kappa_factory = Application, Kappa
Diagram.term_factory = Ty.term_factory = TermBase


Bubble = Box.from_args
