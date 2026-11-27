# -*- coding: utf-8 -*-

"""
The free biclosed monoidal category, i.e. with left and right exponentials.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Exp
    Over
    Under
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
    Functor
    CMap
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Sampler
    Constraints
    AlphaEquation

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, z << y), Box('g', x @ y, z), Box('h', y, x >> z)

>>> Equation(f.uncurry(left=True).curry(left=True), f).draw(
...     doctest='docs/_static/biclosed/curry-left.svg', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-left.svg
    :align: center

>>> Equation(h.uncurry(left=False).curry(left=False), h).draw(
...     doctest='docs/_static/biclosed/curry-right.svg', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-right.svg
    :align: center

>>> Equation(
...     g.curry(left=True).uncurry(left=True), g,
...     g.curry(left=False).uncurry(left=False)).draw(
...         doctest='docs/_static/biclosed/uncurry.svg')

.. image:: /_static/biclosed/uncurry.svg
    :align: center

Compact currying
----------------

:meth:`Diagram.to_compact` bends curry bubbles into coevaluation and feedback,
which lands in :class:`CMap` as a biclosed category has no trace.

>>> g.curry(left=True).uncurry(left=True).to_compact().draw(show=False,
...     doctest="docs/_static/cmap/biclosed-curry-left.dot")

.. graphviz:: /_static/cmap/biclosed-curry-left.dot
    :align: center

>>> g.curry(left=False).uncurry(left=False).to_compact().draw(show=False,
...     doctest="docs/_static/cmap/biclosed-curry-right.dot")

.. graphviz:: /_static/cmap/biclosed-curry-right.dot
    :align: center

"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, replace
from functools import cached_property
from inspect import signature
from itertools import count
from typing import (
    TYPE_CHECKING, Callable, ClassVar, Iterator, Self, Sequence)

from discopy import cat, monoidal, cmap, hypergraph, messages
from discopy.abc import (
    BiclosedCategory, DaggerCategory, DaggerMonoidalCategory, MonoidalCategory,
    ResiduatedMonoid)
from discopy.testable import (
    GENERATORS, Equation as AbstractEquation, Rule, Serialisable,
    Testable, TestableGeneric, axiom)
from discopy.drawing import Drawing
from discopy.cat import AlphaEquivalence, Renamed, factory
# pylint: disable-next=unused-import  # still imported from this module
from discopy.cat import Canonical  # noqa: F401
from discopy.utils import (
    export,
    AxiomError,
    assert_isinstance,
    deprecated_alias,
    factory_key,
    factory_name,
    from_tree,
    get_origin,
)

if TYPE_CHECKING:
    from hypothesis import strategies as st


@factory
class Ty(ResiduatedMonoid, monoidal.Ty):
    """
    A biclosed type is a monoidal type that can be exponentiated.

    Parameters:
        inside (Ty) : The objects inside the type.

    Note
    ----
    Applying a biclosed type to a callable yields a :class:`Abstraction`,
    applying it to a string yields a :class:`Constant`.
    """
    Variable: ClassVar[type["Variable"]]
    Constant: ClassVar[type["Constant"]]
    Application: ClassVar[Callable[..., "Application"]]
    Abstraction: ClassVar[type["Abstraction"]]
    Over: ClassVar[Callable[..., "Exp"]]
    Under: ClassVar[Callable[..., "Exp"]]

    def __pow__(self, other: "Ty | int") -> Self:
        return self.exp(other) if isinstance(other, Ty)\
            else monoidal.Ty.__pow__(self, other)

    def exp(self, other: "Ty") -> Self:
        """
        The exponential of a type, called with ``**``, is its :meth:`over`,
        the default one of the two: the type itself when the exponent is the
        unit, as is any exponential of the unit.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert x ** y == x << y and str(x ** y) == "(x << y)"
        >>> assert x ** Ty() == x << Ty() == Ty() >> x == x
        """
        return self.over(other)

    def over(self, other: Ty) -> Ty:
        return self.ar(self.Over(self, other)) if other else self

    def under(self, other: Ty) -> Ty:
        return self.ar(self.Under(self, other)) if other else self

    def __lshift__(self, other):
        return self.over(other)

    def __rshift__(self, other):
        return other.under(self)

    def __call__(self, arg):
        # pylint: disable=signature-differs  # a lambda, with no offset
        if isinstance(arg, str):
            return self.Constant(arg, self)
        elif isinstance(arg, Callable):
            parameters = dict(signature(arg).parameters)
            left = False
            if "left" in parameters:
                left_param = parameters.pop("left")
                left = left_param.default
                if not isinstance(left, bool):
                    raise NotImplementedError
            varnames = list(parameters.keys())
            if len(varnames) != 1:
                raise NotImplementedError
            var = self.Variable(varnames[0], self)
            return self.Abstraction(var, arg(var), left)
        raise ValueError

    @classmethod
    def strategy(cls, *, max_leaves=3, **params):
        """
        Generate atomic types and their exponentials, ``max_leaves`` atoms
        at most, or with ``params`` the types the search of the free monoid
        derives within them, e.g. between a given ``dom`` and ``cod``.
        """
        from hypothesis import strategies as st

        if params:
            return super().strategy(**params)

        def exponentials(types):
            return st.builds(lambda x, y: x << y, types, types)\
                | st.builds(lambda x, y: x >> y, types, types)

        return st.recursive(
            cls.generator_strategy(), exponentials, max_leaves=max_leaves)

    def cast_wire(self, x: str | cat.Ob) -> cat.Ob:
        """
        Turn a constructor argument into an atom of the type: any monoidal
        wire, as a rigid type is a biclosed type.

        A name gives a :class:`Wire`, and so do a plain ``cat.Ob`` and a
        :class:`monoidal.Wire` with one colour on both sides, colour kept, as
        dumps, pickles and printed text from before hold them. A pickled
        :class:`Exp` of no side becomes the ``over`` of its base.

        >>> assert Ty(cat.Ob('x')) == Ty(monoidal.Wire('x')) == Ty(Wire('x'))
        """
        x = super().cast_wire(x)
        # pylint: disable-next=unidiomatic-typecheck  # not their subclasses
        if type(x) is monoidal.Wire and x.dom == x.cod:
            return Wire(x.name, x.dom, x.cod)
        # pylint: disable-next=unidiomatic-typecheck  # not its subclasses
        return Exp(x.base, x.exponent) if type(x) is Exp else x

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.Atom is monoidal.Wire:
            self.inside = tuple(map(self.cast_wire, self.inside))

    @property
    def is_exp(self):
        """
        Whether the type is an :class:`Exp` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x ** y).is_exp and (x ** y @ Ty()).is_exp
        """
        return len(self) == 1 and isinstance(self.inside[0], Exp)

    @property
    def is_over(self):
        """
        Whether the type is an :class:`Over` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x << y).is_over and (x << y @ Ty()).is_over
        """
        return len(self) == 1 and isinstance(self.inside[0], Over)

    @property
    def is_under(self):
        """
        Whether the type is an :class:`Under` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x >> y).is_under and (x >> y @ Ty()).is_under
        """
        return len(self) == 1 and isinstance(self.inside[0], Under)

    @property
    def base(self):
        "The base of an exponential type, assumes ``self.is_exp``."
        assert self.is_exp
        return self.inside[0].base

    @property
    def exponent(self):
        "The exponent of an exponential type, assumes ``self.is_exp``."
        assert self.is_exp
        return self.inside[0].exponent


class Wire(monoidal.Wire):
    """
    A biclosed object is a self-dagger :class:`monoidal.Wire`, i.e. its left
    and right colours always match. Exponentials do not interact meaningfully
    with colours, so for now we assume nothing is coloured.
    """
    def dagger(self) -> Wire:
        return self


class Exp(Wire):
    """
    A :code:`base` type to an :code:`exponent` type, either :class:`Over` or
    :class:`Under`: a biclosed exponential with no side is the ``over`` of
    its base, the default one, as :meth:`Ty.exp` is.

    Parameters:
        base : The base type.
        exponent : The exponent type.

    >>> x, y = Ty('x'), Ty('y')
    >>> assert Exp(x, y) == Over(x, y) and Ty(Exp(x, y)) == x ** y
    """

    ob = Ty

    # pylint: disable-next=unused-argument  # passed to __init__
    def __new__(cls, *args, **kwargs):
        base = args[0] if args else kwargs.get("base")
        return super().__new__(
            getattr(base, "Over", cls) if cls is Exp else cls)

    def __init__(self, base: Ty, exponent: Ty):
        assert_isinstance(base, self.ob)
        assert_isinstance(exponent, self.ob)
        self.base, self.exponent = base, exponent
        super().__init__("")
        del self.name

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # deleted by __init__
        "The name of an exponential is its print, computed when first read."
        return str(self)

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and (self.base, self.exponent) == (other.base, other.exponent)

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return self.show(self.base, self.exponent)

    def show(self, base, exponent) -> str:
        """ The print of an exponential, given that of its base and exponent.
        """
        return f"({base} ** {exponent})"

    @property
    def drawing_name(self) -> str:
        """
        The label of an exponential wire, made of the labels of its base and
        exponent, e.g. ``(!x >> y)`` rather than ``(x.bang() >> y)``.
        """
        return self.show(self.base.to_drawing(), self.exponent.to_drawing())

    def atoms(self) -> list:
        """ The atoms of an exponential: those of its base and exponent. """
        return self.base.atoms() + self.exponent.atoms()

    def __repr__(self):
        return factory_name(type(self)) + f"({self.base!r}, {self.exponent!r})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'base': self.base.to_tree(),
            'exponent': self.exponent.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['base'], tree['exponent'])))

    @property
    def left(self):
        return self.exponent if isinstance(self, Under) else self.base

    @property
    def right(self):
        return self.base if isinstance(self, Under) else self.exponent


@Ty.slot
class Over(Exp):
    """
    An :code:`exponent` type over a :code:`base` type, called with :code:`<<`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def show(self, base, exponent) -> str:
        return f"({base} << {exponent})"


@Ty.slot
class Under(Exp):
    """
    A :code:`base` type under an :code:`exponent` type, called with :code:`>>`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def show(self, base, exponent) -> str:
        return f"({exponent} >> {base})"


@factory
class Diagram(monoidal.Diagram, BiclosedCategory):
    """
    A biclosed diagram is a monoidal diagram
    with :class:`Curry` and :class:`Eval` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    Box: ClassVar[type["monoidal.Box"]]
    Curry: ClassVar[type["Curry"]]
    Eval: ClassVar[type["Eval"]]
    Coeval: ClassVar[type["Coeval"]]
    Sum: ClassVar[type["monoidal.Sum"]]
    Functor: ClassVar[type["Functor"]]

    ob = Ty

    def curry(self, n=1, left=True) -> Diagram:
        """
        Wrapper around :class:`Curry` called by :class:`Functor`.

        Parameters:
            n : The number of atomic types to curry.
            left : Whether to curry on the left, i.e. into :class:`Over`,
                or on the right, i.e. into :class:`Under`.

        Currying no type is the identity, as an exponential of the unit is
        its base.

        >>> f = Box('f', Ty('x'), Ty('y'))
        >>> assert f.curry(0) == f
        """
        return self.Curry(self, n, left) if n else self

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> "Eval":
        """
        Wrapper around :class:`Eval` called by :class:`Functor`.

        Parameters:
            base : The base of the exponential type to evaluate.
            exponent : The exponent of the exponential type to evaluate.
            left : Whether to evaluate on the left, i.e. from :class:`Over`,
                or on the right, i.e. from :class:`Under`.

        The evaluation at the unit is the identity.
        """
        if not exponent:
            return cls.id(base)
        return cls.Eval(
            base << exponent if left else exponent >> base)

    def to_compact(self) -> CMap:
        """
        Bend curry bubbles into coevaluation and feedback, which lands in
        :class:`CMap` as a biclosed category has no trace, see
        :meth:`~discopy.cmap.CMap.to_compact`.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f.to_map() >> CMap.ev(z, y).dagger()).trace()
        """
        return self.to_map().to_compact()

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor=Functor)

    dagger_involution = DaggerCategory.dagger_involution.failing(
        messages.CURRY_HAS_NO_DAGGER)
    dagger_involution_monoidal = DaggerCategory.dagger_involution.weaken(
        within=MonoidalCategory)
    dagger_contravariance = DaggerCategory.dagger_contravariance.failing(
        messages.CURRY_HAS_NO_DAGGER)
    dagger_contravariance_monoidal = DaggerCategory.dagger_contravariance\
        .weaken(within=MonoidalCategory)
    dagger_monoidality = DaggerMonoidalCategory.dagger_monoidality.failing(
        messages.CURRY_HAS_NO_DAGGER)
    dagger_monoidality_monoidal = DaggerMonoidalCategory.dagger_monoidality\
        .weaken(within=MonoidalCategory)\
        .failing(messages.NO_PLANAR_NORMAL_FORM)
    currying_left = BiclosedCategory.currying_left.failing(messages.FREE_CURRY)
    currying_right = BiclosedCategory.currying_right.failing(
        messages.FREE_CURRY)
    currying_eta_left = BiclosedCategory.currying_eta_left.failing(
        messages.FREE_CURRY)
    currying_eta_right = BiclosedCategory.currying_eta_right.failing(
        messages.FREE_CURRY)
    currying_naturality_left = BiclosedCategory.currying_naturality_left\
        .failing(messages.FREE_CURRY)
    currying_naturality_right = BiclosedCategory.currying_naturality_right\
        .failing(messages.FREE_CURRY)


@Diagram.slot(BiclosedCategory.ev_left)
class Eval(Diagram.Box):
    """
    The evaluation of an exponential type.

    Parameters:
        x : The exponential type to evaluate.
        left : Whether to evaluate ``x @ x.exponent`` on the left or
            ``x.exponent @ x`` on the right, by default the right for an
            :class:`Under` and the left otherwise, as :meth:`Diagram.ev`.
            An :class:`Over` is evaluated on the left only and an
            :class:`Under` on the right only.
    """
    serialised_attrs = ("x", "left")

    def __init__(self, x: Exp, left=None):
        assert x.is_exp
        self.x = x
        exp = x.inside[0]
        self.left = not isinstance(exp, Under) if left is None else left
        if isinstance(exp, Under if self.left else Over):
            raise AxiomError(messages.WRONG_SIDE.format(
                not self.left, x, self.left))
        dom, cod = (x @ x.exponent, x.base) if self.left\
            else (x.exponent @ x, x.base)
        super().__init__("Eval" + str(x), dom, cod)

    def __setstate__(self, state):
        """ A pickle from before is named after ``x ** y``, renamed here. """
        super().__setstate__(state)
        self.name = "Eval" + str(self.x)

    @classmethod
    # pylint: disable-next=arguments-differ  # drawn with no boundary
    def strategy(cls, *, types=None):
        """
        Generate the evaluation of an exponential of two atomic types,
        ``x << y`` on the left or ``y >> x`` on the right.

        Parameters:
            types : The strategy to draw the types from, see
                :meth:`atomic_strategy`.
        """
        from hypothesis import strategies as st

        atoms = cls.atomic_strategy(types)
        return st.tuples(atoms, atoms, st.booleans()).map(
            lambda args: cls(
                args[0] << args[1] if args[2] else args[1] >> args[0],
                left=args[2]))

    def __repr__(self):
        return f"{factory_name(type(self))}({self.x!r}, left={self.left!r})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)),
                    x=self.x.to_tree(), left=self.left)

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['x']), left=tree['left'])

    def dagger(self) -> Coeval:
        return self.Coeval(self.x, self.left)

    @property
    def rule(self) -> Rule:
        """ The evaluation on the left or on the right, as :attr:`left`. """
        return getattr(self.factory, "ev_left" if self.left else "ev_right")

    @property
    def premises(self) -> dict[str, Ty]:
        """ The base and the exponent of the evaluated type. """
        return {"base": self.x.base, "exponent": self.x.exponent}

    @property
    def drawing_name(self):
        return "<<" if self.left else ">>"


@Diagram.slot
class Coeval(Diagram.Box):
    """
    The coevaluation of an exponential type, i.e. the dagger of :class:`Eval`.

    Parameters:
        x : The exponential type to coevaluate.
        left : Whether to coevaluate on the left or right, as :class:`Eval`.

    Note
    ----
    This is not the unit of the adjunction, which sends ``z`` to
    ``(z @ x) << x``, but the transpose of :class:`Eval`, which needs the
    exponent to be dualisable: a biclosed category has no such morphism
    unless its exponential is read at a reflexive object, see `Zeilberger
    (2016) <https://arxiv.org/abs/1512.06751>`_. It is used by
    :meth:`Curry.to_drawing` and :meth:`Diagram.to_compact`.
    """
    serialised_attrs = ("x", "left")
    drawing_name = "lambda"

    __repr__ = Eval.__repr__
    to_tree = Eval.to_tree
    from_tree = classmethod(Eval.from_tree.__func__)
    strategy = classmethod(Eval.strategy.__func__)

    def __init__(self, x: Exp, left=None):
        assert x.is_exp
        self.x = x
        exp = x.inside[0]
        self.left = not isinstance(exp, Under) if left is None else left
        if isinstance(exp, Under if self.left else Over):
            raise AxiomError(messages.WRONG_SIDE.format(
                not self.left, x, self.left))
        cod, dom = (x @ x.exponent, x.base) if self.left\
            else (x.exponent @ x, x.base)
        super().__init__("Coeval" + str(x), dom, cod)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.name = "Coeval" + str(self.x)

    def dagger(self) -> Eval:
        return self.Eval(self.x, self.left)


@Diagram.slot
class Curry(Diagram.Box):
    """
    The currying of a biclosed diagram.

    Parameters:
        arg : The diagram to curry.
        n : The number of atomic types to curry, at most ``len(arg.dom)``.
        left : Whether to curry on the left or right, default is ``True``.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> print(Curry(Box('f', x @ y, z)))
    Curry(f, 1, True)
    """
    serialised_attrs = ("*args", "n", "left")

    def __init__(self, arg: Diagram, n=1, left=True):
        if not 0 <= n <= len(arg.dom):
            raise AxiomError(messages.NOT_SPLITTABLE.format(n, arg.dom))
        self.n, self.left = n, left
        name = f"Curry({arg}, {n}, {left})"
        if left:
            dom = arg.dom[:len(arg.dom) - n]
            cod = arg.cod << arg.dom[len(arg.dom) - n:]
        else:
            dom, cod = arg.dom[n:], arg.dom[:n] >> arg.cod
        Diagram.Box.__init__(self, name, dom, cod, args=(arg, ),
                             drawing_name="$\\Lambda$")

    def __repr__(self):
        return f"{factory_name(type(self))}({self.arg!r}, " \
            f"n={self.n}, left={self.left!r})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)), arg=self.arg.to_tree(),
                    n=self.n, left=self.left)

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['arg']), n=tree['n'], left=tree['left'])

    def __str__(self):
        return self.name

    def to_drawing(self):
        """
        A curry is drawn as its argument followed by a coevaluation, traced
        over the ``n`` wires it curries; a curry of no wire is drawn as its
        argument, as :meth:`Diagram.curry` of no wire is the diagram itself.
        """
        if not self.n:
            return self.arg.to_drawing()
        f, e = self.arg, self.Coeval(self.cod, left=self.left)
        return (f >> e).to_drawing().trace(self.n, left=not self.left)

    def dagger(self):
        """
        A curry has no dagger, which would uncurry an exponential in the
        domain of a diagram.

        >>> x = Ty('x')
        >>> Box('f', x @ x, x).curry().dagger()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: A curry has no dagger.
        """
        raise AxiomError(messages.CURRY_HAS_NO_DAGGER)


Id = Diagram.id


@Diagram.extend
class Functor(monoidal.Functor):
    """
    A biclosed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.

    Note
    ----
    When the codomain has terms, i.e. its constants are its arrows, the functor
    sends terms to terms rather than evaluating them to morphisms:

    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x = (Y << X)("f"), X("x")
    >>> F = Functor(ob_map={X: Ty("A"), Y: Ty("B")},
    ...             ar_map={f: (Ty("B") << Ty("A"))("g"), x: Ty("A")("a")})
    >>> assert F(f(x)) == F(f)(F(x))

    A :class:`CMap` has no terms, so a term is evaluated into it. Nor has a
    :class:`~discopy.hypergraph.Hypergraph`, nor even :meth:`Diagram.ev`, so a
    term is sent there as the image of its diagram:

    >>> assert f(x).to_hypergraph() == f(x).eval().to_hypergraph()

    A :class:`Drawing` has neither, but a term is drawn as the box it is,
    labelled with its name, inside a diagram as well as on its own: only
    :meth:`TermBase.draw` draws its evaluation.

    >>> assert [str(box) for box in (f(x) @ X).to_drawing().boxes] == [
    ...     str(f(x))]

    The image of a term is checked to have the image of its type:

    >>> Functor(ob_map={X: Ty("A"), Y: Ty("B")}, ar_map={f: Ty("A")("a")})(f)
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: Expected a term of type (B << A) for ...
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, TermBase) and self.cod is not Drawing:
            if issubclass(get_origin(self.cod), hypergraph.Hypergraph)\
                    and not isinstance(other, Constant):
                return self(other.eval())
            if issubclass(get_origin(self.cod.ob), Ty) and issubclass(
                    self.cod.ob.Constant, get_origin(self.cod)):
                return self.map_term(other)
            return other.eval(self)
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls):
                base, exponent = self(other.base), self(other.exponent)
                if hasattr(base, attr):
                    return getattr(base, attr)(exponent)
                if hasattr(self.cod, attr):
                    return getattr(self.cod, attr)(base, exponent)
        if isinstance(other, Curry) and hasattr(self.cod, "curry"):
            n = len(self(other.cod.exponent)) if other.n else 0
            return self.cod.curry(self(other.arg), n, other.left)
        if isinstance(other, Coeval) and hasattr(self.cod, "ev"):
            return self(other.dagger()).dagger()
        if self.cod is Drawing:
            if isinstance(other, Ty) and other.inside == (other, ):
                # Avoid infinite recursion when drawing.
                return self.ob_map[other]
        return super().__call__(other)

    def map_variable(self, variable: Variable, context: dict) -> Variable:
        """
        The image of a variable, of the image of its type, renamed with
        underscores until it differs from the images already in the context.

        Parameters:
            variable : The variable to map.
            context : The images of the variables bound so far.
        """
        name, cod = variable.name, self(variable.cod)
        result = self.cod.ob.Variable(name, cod)
        while result in context.values():
            name += "_"
            result = self.cod.ob.Variable(name, cod)
        return result

    def map_term(
            self, term: "TermBase", context: dict | None = None) -> "TermBase":
        """
        The image of a term, a term in the internal language of the codomain,
        checked to have the image of its type and exactly the images of its
        free variables, in order when the codomain is planar.

        Parameters:
            term : The term to map.
            context : The images of the free variables, fresh ones by default.
        """
        if context is None:
            context = {}
            for variable in term.freevars:
                context[variable] = self.map_variable(variable, context)
        result = term.map(self, context)
        if not isinstance(result, TermBase):
            raise AxiomError(f"Expected a term for {term}, got {result}.")
        if result.cod != self(term.cod):
            raise AxiomError(
                f"Expected a term of type {self(term.cod)} for "
                f"{term}, got {result} of type {result.cod}.")
        expected = [context[v] for v in term.freevars]
        ordered = self.cod.ob.Over is not self.cod.ob.Under
        valid = result.freevars == expected if ordered else (
            set(result.freevars) == set(expected))
        if not valid:
            raise AxiomError(f"Expected context {expected} for {term}, "
                             f"got {result.freevars}.")
        return result


CMap = cmap.CMap[Diagram]


class Reducible[factory](TestableGeneric, tuple):
    """
    A term of the ``factory`` with free variables in place of constants, so
    that beta reduction reads a variable at every head, see
    :class:`Normalising`.
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """ None, a sort of premises being drawn over a category. """
        return ()

    @classmethod
    def strategy(cls, **params) -> "st.SearchStrategy[Reducible]":
        """
        Generate a shape and a naming, with no constants.

        Parameters:
            params : Passed to :meth:`TermBase.shapes`.
        """
        from hypothesis import strategies as st

        category = cls.factory
        return st.builds(
            lambda shape, letters: cls((
                category.generate(*shape, letters, constants=False), )),
            category.shapes(**params), category.namings())


class Normalising(Testable):
    """
    The laws of beta reduction, for terms with a ``normal_form`` reached by
    each of their :attr:`reduction_strategies`, e.g.
    :class:`~discopy.closed.TermBase`.
    """
    reduction_strategies: ClassVar[tuple] = ()

    @axiom
    def normal_form_idempotence(
            cls, terms: Reducible[Self]) -> AbstractEquation:
        """ A normal form is its own normal form. """
        normal_form = terms[0].normal_form()
        return Equation(normal_form.normal_form(), normal_form)

    @axiom
    def normal_form_typing(cls, terms: Reducible[Self]) -> AbstractEquation:
        """ Beta reduction preserves the type of a term. """
        return Equation(terms[0].normal_form().cod, terms[0].cod)

    @axiom
    def normal_form_confluence(
            cls, terms: Reducible[Self]) -> AbstractEquation:
        """
        Every reduction strategy reaches the same normal form, up to the
        names of its bound variables.
        """
        return AlphaEquation(*(
            terms[0].normal_form(strategy=strategy)
            for strategy in cls.reduction_strategies))


class TermBase(Diagram.Box, AlphaEquivalence):
    """
    A term in the internal language of biclosed categories, an
    :class:`~discopy.testable.Equivalence` up to the names of its bound
    variables, see :meth:`alpha_eq`.

    Attributes:
        dom (Ty): The tensor of the types for each free variable.
        cod (Ty): The type of a term, i.e. the codomain of its morphism.
        freevars (Ty): The list of free variables.
        functor (Functor): The functor to evaluate the term, ``id`` by default.

    Note
    ----
    Constant terms can be instantiated from any diagram, if the domain is not
    empty (i.e. the diagram is a process not a state) then the constant is a
    given a function type with the argument coming either the left or right:

    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, f, g = X("x"), (X >> Y)("f"), (Y << X)("g")

    Terms can be the :class:`Application` of a function to an argument from its
    left ``>>`` or right ``<<`` with the type inferred automatically e.g.

    >>> xf, gx = x(f, left=True), g(x)
    >>> assert xf.cod == Y == gx.cod

    Applying a biclosed type to a function yields an :class:`Abstraction` e.g.

    >>> f_, g_ = X(lambda y, left=True: y(f, left=True)), X(lambda y: g(y))
    >>> assert f.cod == f_.cod == X >> Y and g.cod == g_.cod == Y << X

    Terms are required to be linear and planar, they can be drawn as diagrams:

    >>> N, S = Ty("N"), Ty("S")
    >>> Alice, loves, Bob = N("Alice"), ((N >> S) << N)("loves"), N("Bob")
    >>> Alice(loves(Bob), left=True).draw(
    ...     doctest='docs/_static/biclosed/alice-loves-bob.svg',
    ...     margins=(.3, 0), figsize=(5, 4))
    """
    dom: Ty
    cod: Ty
    freevars: list[Variable]
    functor: ClassVar[Functor] = Functor.id(Diagram)

    @abstractmethod
    def eval(self, functor: Functor | None = None) -> BiclosedCategory:
        """
        The evaluation of a :class:`Functor` on a term gives a morphism in its
        codomain. By default, this is the identity functor on the free biclosed
        category, i.e. terms are compiled to diagrams with constants as boxes.
        """

    @abstractmethod
    def map(self, functor: Functor, context: dict) -> "Term":
        """
        The image of a term under a :class:`Functor` whose codomain objects
        are biclosed types, i.e. a term in the codomain's internal language
        rather than a morphism.
        """

    def draw(self, **kwargs):
        """
        Drawing a term by evaluating it in the free biclosed category, whereas
        inside a diagram a term is drawn as one box, see :class:`Functor`.
        """
        return self.eval().draw(**kwargs)

    def to_closed(self):
        """
        Translate the term into the internal language of a closed category.
        """
        from discopy.closed import TermBase
        return TermBase.from_biclosed(self)

    def __call__(self, other, left=False):
        args = (other, self, left) if left else (self, other, left)
        return self.cod.Application(*args)

    @classmethod
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def generate(  # a shape, a naming and whether it takes constants
            cls, cod: Ty, choices: Sequence[int], types: Sequence[Ty],
            letters: Sequence[str], constants: bool = True) -> "Term":
        """
        Build a term of type ``cod`` from a sequence of ``choices``, one per
        node of the term, taking leaves once it runs out; the exponents of
        its applications are drawn from ``types`` and the variable bound at
        each level of binders is named by the letter at that level, so that
        the same choices under other letters give an alpha-equivalent term.
        Free variables are named ``v0, v1, ...`` and constants ``c0, c1, ...``,
        free variables taking the place of constants unless ``constants``.
        The term is planar and linear, as any term of a closed category is
        too, see :class:`Sampler` for the procedure.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> term = TermBase.generate(Y << X, [8, 0, 0], [X], "xy")
        >>> print(term)
        X(lambda x0: (Y << X)('c0')(x0))
        >>> assert term.alpha_eq(
        ...     TermBase.generate(Y << X, [8, 0, 0], [X], "z"))
        """
        return Sampler(
            cls, iter(choices), types, letters, constants=constants).term(cod)

    @classmethod
    def shapes(cls, *, types=None, cod=None) -> st.SearchStrategy[tuple]:
        """
        Generate the arguments of :meth:`generate` but its letters: a type,
        the choices and the types the exponents are drawn from, so that one
        shape under several namings gives :class:`Renamed` terms.
        """
        from hypothesis import strategies as st

        types = cls.ob.strategy() if types is None else types
        cods = types if cod is None else st.just(cod)
        return st.tuples(
            cods,
            cls.choices(),
            st.lists(types, min_size=1, max_size=3))

    strategy = classmethod(AlphaEquivalence.strategy.__func__)

    @axiom
    def alpha_application(cls, terms: Renamed[Self]) -> AbstractEquation:
        """
        Alpha-equivalence is a congruence with respect to application: a
        function applied to alpha-equivalent arguments gives alpha-equivalent
        terms, and so do alpha-equivalent functions applied to an argument.
        """
        x = cls.ob(GENERATORS[0])
        f = ((x << x) << terms[0].cod)("f")
        a = x("a")  # pylint: disable=not-callable  # x is a Ty
        return AlphaEquation(*(f(term)(a) for term in terms))

    @axiom
    def alpha_abstraction(cls, terms: Renamed[Self]) -> AbstractEquation:
        """
        Alpha-equivalence is a congruence with respect to abstraction:
        binding a variable in alpha-equivalent bodies gives alpha-equivalent
        terms.
        """
        x = cls.ob(GENERATORS[0])
        f, w = ((x << x) << terms[0].cod)("f"), cls.ob.Variable("w", x)
        return AlphaEquation(*(
            cls.ob.Abstraction(w, f(term)(w), False)
            for term in terms))


@Ty.slot
class Constant(TermBase):
    """
    A constant term with a string as name and a :class:`Ty`, i.e. a term with
    no free variable: ``X("x")`` is a constant of type ``X`` and
    ``(X >> Y)("f")`` a function constant, applied to an argument by
    :class:`Application`. Under the identity functor it is a box with the
    empty domain, i.e. a state of its type; any other :class:`Functor`
    interprets it through its :code:`ar_map`.

    Attributes:
        name (str): The name of the constant.
        cod (Ty): The type of the constant.
    """
    serialised_attrs = ("name", "cod", "*", "is_dagger", "data")

    def __init__(self, name: str, cod: Ty, **kwargs):
        super().__init__(name, dom=self.ob.id(cod.dom), cod=cod, **kwargs)
        self.freevars = []

    @classmethod
    def from_tree(cls, tree):
        kwargs = {k: tree[k] for k in ('data', 'is_dagger') if k in tree}
        return cls(tree['name'], from_tree(tree['cod']), **kwargs)

    @property
    def constants(self):
        return [self]

    def atoms(self) -> list:
        """ A constant prints as its type called on its name: its atoms. """
        return self.cod.atoms()

    def eval(self, functor=None):
        return cat.Functor.__call__(functor or self.functor, self)

    def map(self, functor, context):
        return functor.ar_map[self]

    def __repr__(self):
        data = "" if self.data is None else f", data={self.data!r}"
        return factory_name(type(self))\
            + f"({self.name!r}, {self.cod!r}{data})"

    def __str__(self):
        return f"{self.cod!s}({self.name!r})"


@Ty.slot
class Variable(TermBase):
    """
    A variable with a string as name and a :class:`Ty`.

    Attributes:
        name (str): The name of the variable
        cod (Ty): The type of the variable.
    """
    serialised_attrs = ("name", "cod")

    def __init__(self, name: str, cod: Ty):
        super().__init__(name, dom=cod, cod=cod)
        self.freevars = [self]

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['name'], from_tree(tree['cod']))

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.cod.id(functor(self.cod))

    def map(self, functor, context):
        return context[self]

    @property
    def constants(self):
        return []

    def atoms(self) -> list[Variable]:
        """ A variable prints as its bare name, it is its own atom. """
        return [self]

    alpha_eq_under = cat.Variable.alpha_eq_under
    fresh = classmethod(cat.Variable.fresh.__func__)

    __repr__ = Constant.__repr__


@Ty.slot
class Application(TermBase):
    """
    The application either ``function(argument)`` of a term ``function`` of
    type ``Y << X`` to a term ``argument`` of type ``X`` or
    ``argument(function, left=True)`` of a term ``argument`` of type ``X`` fed
    as input to a term ``function`` of type ``X >> Y``.

    Attributes:
        function (Term): The function being applied.
        argument (Term): The argument to which the function is applied.
        left (bool): Whether the argument comes in from the left or right.
    """
    serialised_attrs = ("function", "argument", "left")

    def __init__(
            self, function: TermBase, argument: TermBase, left: bool = False):
        assert_isinstance(function, TermBase)
        assert_isinstance(argument, TermBase)
        if not function.cod.is_exp:
            raise TypeError(
                messages.NOT_EXPONENTIAL.format(function, function.cod))
        self.function, self.argument, self.left = function, argument, left
        if function.cod.exponent != argument.cod:
            raise ValueError(
                f"Expected {function.cod.exponent}, got {argument.cod}")
        cod, fname, xname = function.cod.base, str(function), str(argument)
        name = f"{xname}({fname}, left=True)" if left else f"{fname}({xname})"
        dom = self.__check_dom__(function, argument, left)
        super().__init__(name, dom, cod)

    def __check_dom__(self, func, args, left):
        assert_isinstance(func.cod.inside[0], Under if left else Over)
        if set(func.freevars).intersection(args.freevars):
            raise ValueError("Expected disjoint free variables.")
        self.freevars = args.freevars + func.freevars if self.left\
            else func.freevars + args.freevars
        return args.dom @ func.dom if left else func.dom @ args.dom

    def eval(self, functor=None):
        functor = functor or self.functor
        func = self.function.eval(functor=functor)
        args = self.argument.eval(functor=functor)
        base, exponent = self.function.cod.base, self.function.cod.exponent
        ev = functor.cod.ev(
            functor(base), functor(exponent), left=not self.left)
        return args @ func >> ev if self.left else func @ args >> ev

    def map(self, functor, context):
        return functor.cod.ob.Application(
            functor.map_term(self.function, context),
            functor.map_term(self.argument, context), self.left)

    def __repr__(self):
        func, arg = repr(self.function), repr(self.argument)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({func}, {arg}{left})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)),
                    function=self.function.to_tree(),
                    argument=self.argument.to_tree(), left=self.left)

    @staticmethod
    def subtrees(tree):
        """
        The trees of the function and the argument of an application, read
        from the keys ``func`` and ``args`` of a tree from before they were
        spelled out.
        """
        return tuple(map(from_tree, (
            tree.get('function', tree.get('func')),
            tree.get('argument', tree.get('args')))))

    @classmethod
    def from_tree(cls, tree):
        return cls(*cls.subtrees(tree), left=tree['left'])

    def setoid(self):
        """
        Applications are equal when their functions, arguments and sides
        are, rather than when they print the same between the same types:
        a closed constant prints as its bare name, whatever its type.
        """
        return factory_key(type(self)), self.function, self.argument, self.left

    @property
    def constants(self):
        return self.argument.constants + self.function.constants if self.left\
            else self.function.constants + self.argument.constants

    def atoms(self) -> list:
        """ The atoms of an application: those of its function and args. """
        return self.function.atoms() + self.argument.atoms()

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Applications on one side, functions and arguments alike. """
        if any(type(other) is not type(self) or other.left != self.left
               for other in others):
            return False
        return self.function.alpha_eq_under(
            scopes, [other.function for other in others], depth)\
            and self.argument.alpha_eq_under(
                scopes, [other.argument for other in others], depth)


@Ty.slot
class Abstraction(TermBase):
    serialised_attrs = ("var", "body", "left")
    var: Variable
    body: TermBase
    left: bool = False

    def __init__(self, var: Variable, body: TermBase, left: bool = False):
        self.var, self.body, self.left = var, body, left
        left_str = ", left=True" if left else ""
        name = f"{var.cod}(lambda {var.name}{left_str}: {body})"
        cod = var.cod >> body.cod if left else body.cod << var.cod
        dom = self.__check_dom__()
        super().__init__(name, dom, cod)

    def __check_dom__(self):
        body_freevars = self.body.freevars
        if body_freevars.count(self.var) != 1:
            raise ValueError("Expected variable to occur exactly once.")
        index = body_freevars.index(self.var)
        if self.left and index != 0:
            raise ValueError("Expected abstraction of left-most variable.")
        if not self.left and index != len(body_freevars) - 1:
            raise ValueError("Expected abstraction of right-most variable.")
        self.freevars = body_freevars[1:] if self.left else body_freevars[:-1]
        n = len(self.var.cod)
        return self.body.dom[n:] if self.left\
            else self.body.dom[:len(self.body.dom) - n]

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.cod.curry(
            self.body.eval(functor),  # ty: ignore[invalid-argument-type]
            len(functor(self.var.cod)), not self.left)

    def map(self, functor, context):
        context = {key: value for key, value in context.items()
                   if key != self.var}
        variable = functor.map_variable(self.var, context)
        context[self.var] = variable
        return functor.cod.ob.Abstraction(
            variable, functor.map_term(self.body, context), self.left)

    def __repr__(self):
        var, body = repr(self.var), repr(self.body)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({var}, {body}{left})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)), var=self.var.to_tree(),
                    body=self.body.to_tree(), left=self.left)

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['var'], tree['body'])),
                   left=tree['left'])

    def setoid(self):
        """
        Abstractions are equal when their variables, bodies and sides are,
        see :meth:`Application.setoid`.
        """
        return factory_key(type(self)), self.var, self.body, self.left

    @property
    def constants(self):
        return self.body.constants

    def atoms(self) -> list:
        """
        The atoms of an abstraction: those of the type of its variable and
        of its body but the variable, which the lambda it prints binds.
        """
        return self.var.cod.atoms() + [
            atom for atom in self.body.atoms() if atom != self.var]

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Binders of one type on one side, bodies alike one level down. """
        if any(type(other) is not type(self)
               or (other.left, other.var.cod) != (self.left, self.var.cod)
               for other in others):
            return False
        terms = (self, *others)
        return self.alpha_eq_bound(
            scopes, [term.body for term in terms],
            [[term.var] for term in terms], depth)


type Term = Constant | Variable | Application | Abstraction


for law in sorted(
        set(Diagram.Box.axioms) - set(Serialisable.axioms)
        | set(Diagram.Box.rules)):
    setattr(TermBase, law, getattr(Diagram.Box, law).inapplicable(
        messages.TERM_IS_A_BOX))


@dataclass
class Sampler:
    """
    Samples a term from a sequence of choices, one per node, taking leaves
    once it runs out, see :meth:`TermBase.generate`.

    The term is planar and linear: the free variables of each subterm open
    with a prefix and close with a suffix of the bound variables in scope,
    with extra free ones in between or not, its :class:`Constraints`, which
    an application :meth:`splits` between its function and its argument and
    an :meth:`abstraction` extends with the variable it binds; a subterm that
    cannot be a leaf is a :meth:`spine`, a constant applied to the bound
    variables in order. The options of a choice are closures calling the
    method that builds the node with its arguments spelt out, so that the
    recursion runs in Python frames, as deep as the recursion limit allows:
    :func:`functools.partial`, or a call through ``*args``, goes through C
    at every level and overflows its stack a few thousand nodes deep.

    Parameters:
        category : The class of terms to sample.
        choices : The choices left, consumed one per node.
        types : The types the exponents of applications are drawn from.
        letters : The letters naming the bound variables, one per level.
        counter : The numbers of the free variables and constants built.
        linear : Whether each bound variable occurs exactly once, in the
            order of the binders, as :mod:`biclosed` requires; a term of
            :mod:`~discopy.closed` may use one any number of times.
        constants : Whether a leaf or the head of a spine may be a
            constant, else a free variable, so that beta reduction reads a
            variable at every head, see :class:`Reducible`.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> sampler = Sampler(TermBase, iter([]), [X], "x")
    >>> print(sampler.spine(Y, (Variable("x0", X), )))
    (Y << X)('c0')(x0)
    >>> print(sampler.term(Y << X))
    (Y << X)('c1')
    """
    category: type[TermBase]
    choices: Iterator[int]
    types: Sequence[Ty]
    letters: Sequence[str]
    counter: Iterator[int] = field(default_factory=count)
    linear: bool = True
    constants: bool = True

    def choose(self, options: Sequence):
        """ Pick an option by the next choice, the first when they ran out. """
        return options[next(self.choices, 0) % len(options)]

    def constant(self, cod: Ty) -> Constant | Variable:
        """ A fresh constant of a given type, if the sampler takes any. """
        if not self.constants:
            return self.variable(cod)
        return self.category.ob.Constant(
            f"c{next(self.counter)}", cod)

    def variable(self, cod: Ty) -> Variable:
        """ A fresh free variable of a given type. """
        return self.category.ob.Variable(
            f"v{next(self.counter)}", cod)

    def bound(self, level: int, cod: Ty) -> Variable:
        """ The variable bound at a given level, named by its letter. """
        letter = self.letters[level % len(self.letters)]
        return self.category.ob.Variable(f"{letter}{level}", cod)

    def spine(self, cod: Ty, bound: tuple[Variable, ...]) -> Term:
        """ A fresh constant applied to the bound variables in order. """
        function_type = cod
        for variable in reversed(bound):
            function_type = function_type << variable.cod
        result = self.constant(function_type)
        for variable in bound:
            result = self.category.ob.Application(
                result, variable, False)
        return result

    @staticmethod
    def leaf(variable: Variable) -> Callable[[], Variable]:
        """ The option of a bound variable as a leaf. """
        return lambda: variable

    def leaves(self, cod: Ty, bound: tuple[Variable, ...],
               extra: bool) -> list[Callable[[], Term]]:
        """
        The leaves allowed: a constant or, if ``extra``, a free variable
        when no variable is bound, and the bound variables of the type; a
        linear term takes the only bound variable and nothing else.
        """
        result = []
        if not bound or not self.linear:
            result.append(lambda: self.constant(cod))
        if (not bound or not self.linear) and extra:
            result.append(lambda: self.variable(cod))
        if not self.linear or len(bound) == 1:
            result += [self.leaf(variable)
                       for variable in bound if variable.cod == cod]
        return result

    def splits(self, constraints: Constraints
               ) -> list[tuple[Constraints, Constraints]]:
        """
        The ways of splitting the constraints between two subterms in
        sequence: at a bound variable of the prefix, among the extras or at
        a bound variable of the suffix, each a pair of constraints; a term
        that need not be linear passes every bound variable to both.
        """
        if not self.linear:
            return [(constraints, constraints)]
        prefix, suffix = constraints.prefix, constraints.suffix
        result = [(
            replace(constraints, prefix=prefix[:i], suffix=(), extra=False),
            replace(constraints, prefix=prefix[i:]))
            for i in range(len(prefix) + 1)]
        result += [(
            replace(constraints, suffix=()),
            replace(constraints, prefix=()))] if constraints.extra else []
        result += [(
            replace(constraints, suffix=suffix[:j]),
            replace(constraints, prefix=(), suffix=suffix[j:], extra=False))
            for j in range(len(suffix) + 1)]
        return result

    def application(self, cod: Ty, first: Constraints, second: Constraints,
                    left: bool) -> Application:
        """
        A function applied to an argument of an exponent drawn from the
        types, ``first`` the constraints of whichever comes first in the
        free variables: the function, or the argument if ``left``.
        """
        exponent = self.choose(self.types)
        if left:
            args = self.term(exponent, first)
            func = self.term(exponent >> cod, second)
            return self.category.ob.Application(func, args, True)
        func = self.term(cod << exponent, first)
        args = self.term(exponent, second)
        return self.category.ob.Application(func, args, False)

    def applications(self, cod: Ty, constraints: Constraints
                     ) -> list[Callable[[], Term]]:
        """ The applications allowed, one per split and side. """
        return [self.split(cod, first, second, left)
                for first, second in self.splits(constraints)
                for left in (False, True)]

    def split(self, cod: Ty, first: Constraints, second: Constraints,
              left: bool) -> Callable[[], Application]:
        """ The option of an application with its constraints split. """
        return lambda: self.application(cod, first, second, left)

    def abstraction(self, cod: Ty, constraints: Constraints) -> Abstraction:
        """
        The abstraction of the variable bound at this level, first in the
        free variables of the body when the type is a left exponential and
        last otherwise.
        """
        left = cod.is_under
        var = self.bound(constraints.level, cod.exponent)
        opening, closing = ((var, *constraints.prefix), constraints.suffix)\
            if left else (constraints.prefix, (*constraints.suffix, var))
        body = self.term(cod.base, replace(
            constraints, prefix=opening, suffix=closing,
            level=constraints.level + 1))
        return self.category.ob.Abstraction(var, body, left)

    def term(self, cod: Ty, constraints: "Constraints | None" = None) -> Term:
        """
        A term of a given type under the constraints, none by default: a
        leaf when the choices ran out, or the spine when none fits, else the
        option the next choice picks among the leaves, the applications and
        the abstraction if the type is an exponential.
        """
        constraints = Constraints() if constraints is None else constraints
        bound = constraints.variables
        options = self.leaves(cod, bound, constraints.extra)
        choice = next(self.choices, None)
        if choice is None:
            return options[0]() if options else self.spine(cod, bound)
        options += self.applications(cod, constraints)
        if cod.is_exp:
            options.append(lambda: self.abstraction(cod, constraints))
        return options[choice % len(options)]()


@dataclass(frozen=True)
class Constraints:
    """
    The constraints on a subterm of a :class:`Sampler`: its free variables
    open with a ``prefix`` and close with a ``suffix`` of the bound
    variables in scope, with ``extra`` free ones in between or not, and its
    binders are named from ``level``.

    Example
    -------
    >>> X = Ty("X")
    >>> x = Variable("x0", X)
    >>> assert Constraints((x, )).variables == (x, ) and Constraints().extra
    """
    prefix: tuple[Variable, ...] = ()
    suffix: tuple[Variable, ...] = ()
    extra: bool = True
    level: int = 0

    @property
    def variables(self) -> tuple[Variable, ...]:
        """ The bound variables in scope, the prefix then the suffix. """
        return self.prefix + self.suffix


class Equation(monoidal.Equation):
    """ The :class:`monoidal.Equation` of biclosed diagrams. """


class AlphaEquation(cat.AlphaEquation, Equation):
    """
    An :class:`Equation` between terms which holds when they are
    alpha-equivalent, see :meth:`~discopy.cat.AlphaEquivalence.alpha_eq`.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> f = (Y << X)("f")
    >>> assert AlphaEquation(X(lambda x: f(x)), X(lambda y: f(y)))
    >>> assert not AlphaEquation(f(Variable("x", X)), f(Variable("y", X)))
    """


TermBase.Equation = AlphaEquation

__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})


Bubble = Diagram.Box.from_args


export(__name__)
