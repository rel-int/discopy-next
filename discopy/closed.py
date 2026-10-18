
"""
The free closed markov category, i.e. with copy, discard, exponentials and
products that are not strictly associative.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Product
    Unitype
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Compound
    Tuple
    Projection
    Let
    Substitution
    BohmTree
    Strategy
    LeftmostOutermost
    RightmostFirst
    Diagram
    Box
    Eval
    Coeval
    Curry
    Pack
    Unpack
    Discard
    Sum
    Functor
    CMap

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        let

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', x, z << y), Box('g', x @ y, z)

>>> Equation(f.uncurry().curry(), f).draw(
...     doctest='docs/_static/closed/curry-left.svg', margins=(0.1, 0.05))

.. image:: /_static/closed/curry-left.svg
    :align: center

>>> Equation(g.curry().uncurry(), g).draw(
...     doctest='docs/_static/closed/uncurry.svg')

.. image:: /_static/closed/uncurry.svg
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property, reduce
from inspect import signature
from typing import Callable, ClassVar, Dict, Self

from discopy import monoidal, biclosed, markov, cmap, hypergraph, messages
from discopy.abc import ClosedCategory, MonoidalCategory
from discopy.axioms import Serialisable
from discopy.cat import factory
from discopy.drawing import Drawing
from discopy.utils import (
    AxiomError, assert_isinstance, factory_name, from_tree)


@factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.

    Applying a closed type to a function yields an :class:`Term` e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> t = X(lambda x: (X >> Y)(lambda f: f(x)))
    >>> t.draw(
    ...     doctest='docs/_static/closed/diagram.svg',
    ...     aspect="auto", figsize=(8, 8), margins=(0.2, 0))

    .. image:: /_static/closed/diagram.svg
        :align: center
    """
    product_factory: ClassVar[type["Product"]]
    variable_factory: ClassVar[type["Variable"]]
    constant_factory: ClassVar[type["Constant"]]
    application_factory: ClassVar[type["Application"]]
    abstraction_factory: ClassVar[type["Abstraction"]]

    def __mul__(self, other: Ty) -> Ty:
        return self.product(other)

    def product(self, *others: Ty) -> Ty:
        "The :class:`Product` of a type with a tuple of ``others``."
        return self.ar(self.product_factory(self, *others))

    @property
    def is_product(self):
        """
        Whether the type is a :class:`Product` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x * y).is_product and (x * y @ Ty()).is_product
        """
        return len(self) == 1 and isinstance(self.inside[0], Product)

    @property
    def factors(self):
        "The factors of a product type, assumes ``self.is_product``."
        assert self.is_product
        return self.inside[0].factors

    @classmethod
    def from_biclosed(cls, old: biclosed.Ty) -> Self:
        """
        Translate a biclosed type into a closed type, collapsing left and
        right exponentials into a single exponential.

        Parameters:
            old : The biclosed type to translate.

        Example
        -------
        >>> x, y = biclosed.Ty("x"), biclosed.Ty("y")
        >>> assert Ty.from_biclosed(x << y) == Ty.from_biclosed(y >> x)
        """
        return biclosed.Functor(
            ob_map=lambda x: cls(x.inside[0].name),
            cod=cls.constant_factory.functor.cod)(old)


class Exp(biclosed.Exp):
    "An exponential object in a markov category."

    ob = Ty

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


class Product(biclosed.Wire):
    """
    The product of a tuple of types, which is not strictly associative,
    called with ``*`` in the binary case.

    Parameters:
        factors : The factors of the product.

    Example
    -------
    >>> X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    >>> assert X * Y == Ty(Product(X, Y))
    >>> assert (X * Y) * Z != X * (Y * Z) != X.product(Y, Z)

    Evaluation strictifies a product to the tensor of its factors, see
    :class:`Pack`, so that the three types above are all interpreted as
    ``X @ Y @ Z``.
    """
    ob = Ty

    def __init__(self, *factors: biclosed.Ty):
        for typ in factors:
            assert_isinstance(typ, self.ob)
        self.factors = factors
        super().__init__(str(self))

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and self.factors == other.factors

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        if len(self.factors) == 2:
            return "(" + " * ".join(
                str(typ) if len(typ) == 1 else f"({typ})"
                for typ in self.factors) + ")"
        if not self.factors:
            return f"{type(self).__name__}()"
        first, *others = self.factors
        first = str(first) if len(first) == 1 else f"({first})"
        return f"{first}.product({', '.join(map(str, others))})"

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.factors))})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'factors': [typ.to_tree() for typ in self.factors]}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, tree['factors']))


class Unitype(Exp):
    """
    The unitype is its own exponential, i.e. ``U == U >> U``: embedding the
    untyped lambda calculus, where every term can be applied to every other.

    Example
    -------
    >>> U = Ty(Unitype())
    >>> assert U == U >> U and U.base == U == U.exponent
    """
    def __init__(self):
        monoidal.Wire.__init__(self, "U")
        self.base = self.exponent = Ty(self)

    def __eq__(self, other):
        return isinstance(other, Unitype) or isinstance(other, Exp)\
            and (other.base, other.exponent) == (self.base, self.exponent)

    def __hash__(self):
        return hash(repr(Exp(self.base, self.exponent)))

    def __str__(self):
        return "U"

    def __repr__(self):
        return "Unitype()"

    def to_tree(self):
        return {'factory': factory_name(type(self))}

    @classmethod
    def from_tree(cls, tree):
        # pylint: disable=unused-argument  # the unitype has no parameter
        return cls()


@factory
class Diagram(markov.Diagram, biclosed.Diagram, ClosedCategory):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    pack_factory: ClassVar[type["Pack"]]
    unpack_factory: ClassVar[type["Unpack"]]
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    copy_factory: ClassVar[type["Copy"]]
    merge_factory: ClassVar[type["Merge"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    curry_factory: ClassVar[type["Curry"]]
    eval_factory: ClassVar[type["Eval"]]
    coeval_factory: ClassVar[type["Coeval"]]
    trace_factory: ClassVar[type["Trace"]]
    discard_factory: ClassVar[type["Discard"]]
    sum_factory: ClassVar[type["Sum"]]
    ob = Ty

    @property
    def is_linear(self):
        return all(box.is_linear for box in self.boxes)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True):
        if not exponent:
            return cls.id(base)
        return cls.eval_factory(exponent >> base, left=left)

    def to_compact(self) -> Diagram:
        """
        Open the curry bubbles into coevaluation and feedback, which stays
        a :class:`Diagram` as a closed category is traced: each curry
        becomes its argument followed by :class:`Coeval`, traced over the
        curried wires, and each term is evaluated first.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f >> Coeval(z << y, left=True)).trace()
        """
        def image(box):
            if isinstance(box, Curry):
                return (box.arg.to_compact() >> Coeval(
                    box.cod, left=box.left)).trace(
                        len(box.cod.exponent), left=not box.left)
            if isinstance(box, (
                    Application, Abstraction, Tuple, Projection, Let)):
                return box.eval(Functor.id(Diagram)).to_compact()
            return box
        result = self.id(self.dom)
        for layer in self.inside:
            for box, offset in layer.boxes_and_offsets:
                cod = result.cod
                result >>= cod[:offset] @ image(box)\
                    @ cod[offset + len(box.dom):]
        return result

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    dagger_monoidality = ClosedCategory.dagger_monoidality.failing(
        messages.CURRY_HAS_NO_DAGGER)
    dagger_monoidality_monoidal = ClosedCategory.dagger_monoidality.weaken(
        within=MonoidalCategory)
    currying_symmetry = ClosedCategory.currying_symmetry.failing(
        messages.FREE_CURRY)

    def to_term(self) -> Term:
        """
        Read a causal diagram as a term in fine-grain call-by-value style:
        one let statement per box in topological order with a variable for
        each wire, going through :class:`Hypergraph` so that the copy,
        discard and swap structure simplifies away into the spiders.

        The inputs of the diagram are the variables ``x0, ..., x{n - 1}``
        in the order of ``dom``. The free variables of the term are the
        inputs that the diagram actually uses, in the order of their first
        occurrence in the term rather than that of ``dom``, so that
        :meth:`TermBase.eval` gives a diagram with the used inputs in that
        order as domain; pass the inputs as its ``context`` to get ``dom``
        back. A diagram with no box is a tuple of variables.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> f, g = Box("f", X, Y @ Y), Box("g", Y @ Y, Y)
        >>> diagram = Diagram.copy(X) >> f @ Diagram.discard(X) >> g
        >>> print(diagram.to_term())
        let(f(x0), lambda x1, x2: g(Tuple(x1, x2)))
        >>> assert Diagram.swap(X, Y).to_term()\\
        ...     == Tuple(Variable("x1", Y), Variable("x0", X))

        A swap before a box puts its inputs in the order the box reads them:

        >>> h = Box("h", Y @ X, X)
        >>> term = (Diagram.swap(X, Y) >> h).to_term()
        >>> x0, x1 = Variable("x0", X), Variable("x1", Y)
        >>> assert term.freevars == [x1, x0] and term.eval().dom == Y @ X
        >>> assert term.eval(context=[x0, x1]).dom == X @ Y
        """
        graph = Hypergraph.from_diagram(self)
        if not graph.is_causal:
            raise ValueError(f"Expected a causal diagram, got {self}")
        variables = [self.ob.variable_factory(f"x{i}", typ)
                     for i, typ in enumerate(graph.spider_types)]
        outputs = [variables[i] for i in graph.cod_wires]
        result = outputs[0] if len(outputs) == 1 else Tuple(*outputs)
        for box, (dom_wires, cod_wires) in reversed(list(zip(
                graph.boxes, graph.box_wires))):
            expression = self.box_to_term(
                box, [variables[i] for i in dom_wires])
            bound = [variables[i] for i in cod_wires]
            trivial = bound == [result] or not bound\
                and isinstance(result, Tuple) and not result.terms
            result = expression if trivial\
                else Let(expression, tuple(bound), result)
        return result

    @classmethod
    def box_to_term(cls, box, args):
        "The application of a box as a constant to variables for its inputs."
        if not box.dom:
            return box.cod(box.name)
        factors = [box.dom[i:i + 1] for i in range(len(box.dom))]
        exponent = box.dom if len(factors) == 1\
            else factors[0].product(*factors[1:])
        constant = (exponent >> box.cod)(box.name)
        return constant(args[0] if len(args) == 1 else Tuple(*args))


class Box(markov.Box, biclosed.Box, Diagram):
    "A closed box is a markov and biclosed box in a closed diagram."
    is_linear = True


class Eval(biclosed.Eval, Box):
    "The evaluation of an exponential type."
    drawing_name = "__call__"


class Coeval(biclosed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."


class Curry(biclosed.Curry, Box):
    "The currying of a closed diagram, linear when its argument is."

    @property
    def is_linear(self):
        return self.arg.is_linear


class Pack(Box):
    """
    The canonical isomorphism from the tensor of the factors of a
    :class:`Product` type to the product itself.

    Parameters:
        cod : The product type to pack into.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert Pack(X * Y).dom == X @ Y and Pack(X * Y).cod == X * Y
    >>> assert Pack(X * Y).dagger() == Unpack(X * Y)
    """
    def __init__(self, cod: biclosed.Ty):
        if not cod.is_product:
            raise TypeError(f"Expected {Product}, got {cod!r}")
        dom = self.ob().tensor(*cod.factors)
        super().__init__(f"Pack({cod})", dom, cod)

    def dagger(self):
        return Unpack(self.cod)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.cod!r})"

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['cod']))


class Unpack(Box):
    """
    The canonical isomorphism from a :class:`Product` type to the tensor of
    its factors, i.e. the dagger of :class:`Pack`.

    Parameters:
        dom : The product type to unpack.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert Unpack(X * Y).dom == X * Y and Unpack(X * Y).cod == X @ Y
    >>> assert Unpack(X * Y).dagger() == Pack(X * Y)
    """
    def __init__(self, dom: biclosed.Ty):
        if not dom.is_product:
            raise TypeError(f"Expected {Product}, got {dom!r}")
        cod = self.ob().tensor(*dom.factors)
        super().__init__(f"Unpack({dom})", dom, cod)

    def dagger(self):
        return Pack(self.dom)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.dom!r})"

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'dom': self.dom.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['dom']))


class Permutation(markov.Permutation, Box):
    "A permutation in a closed diagram."


class Swap(Permutation, markov.Swap, Box):
    "Symmetric swap in a closed diagram."


class Trace(markov.Trace, Box):
    "A trace in a closed category, linear when its argument is."

    @property
    def is_linear(self):
        return self.arg.is_linear


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"

    is_linear = False


class Discard(markov.Discard, Copy):
    "A markov discard in a closed category."


class Merge(markov.Merge, Box):
    """
    A markov merge in a closed category, the dagger of a copy.

    >>> x = Ty('x')
    >>> assert Copy(x).dagger() == Merge(x)
    >>> assert Diagram.discard(x).dagger().dagger() == Discard(x)
    """

    is_linear = False


class Sum(markov.Sum, biclosed.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box,
    linear when every term is.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    @property
    def is_linear(self):
        return all(term.is_linear for term in self.terms)


class Functor(biclosed.Functor, markov.Functor):
    """
    A closed functor is a markov functor that preserves evaluation, currying
    and packing. When the codomain has no products, i.e. its objects have no
    ``product`` method, the functor strictifies: a :class:`Product` is mapped
    to the tensor of its factors and :class:`Pack`, :class:`Unpack` to the
    identity. The exception is :class:`Drawing` where a product is drawn as
    a single wire and its packing as a box.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if self.cod is Drawing:
            return super().__call__(other)
        if isinstance(other, Product):
            if hasattr(self.cod.ob, "product"):
                return self.cod.ob(self.cod.ob.product_factory(
                    *map(self, other.factors)))
            return self(self.dom.ob().tensor(*other.factors))
        if isinstance(other, (Pack, Unpack)):
            typ = other.cod if isinstance(other, Pack) else other.dom
            if not hasattr(self.cod.ob, "product"):
                return self.cod.id(self(typ))
            if hasattr(self.cod, "pack_factory"):
                box = self.cod.pack_factory if isinstance(other, Pack)\
                    else self.cod.unpack_factory
                return box(self(typ))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]


Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.pack_factory = Pack
Diagram.unpack_factory = Unpack
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.trace_factory = Trace
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)

Id = Diagram.id


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a closed category, i.e. a lambda term
    which need not be linear: this module implements closed markov categories
    by design, so a variable may be copied and discarded, i.e. occur any
    number of times.

    A term is evaluated in a context, a list of distinct variables containing
    its free ones: the variables that do not occur in the term are discarded,
    the others are permuted into the order of its :attr:`freevars`, see
    :meth:`weaken`. The context is :attr:`freevars` itself by default.

    Note
    ----
    Closed terms accept the ``left`` argument of their biclosed counterparts
    and ignore it: a closed category has one exponential, so an application
    ``x(f, left=True)`` is ``f(x)`` and an abstraction on the left is one on
    the right.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x = (X >> Y)("f"), X("x")
    >>> assert x(f, left=True) == f(x)
    """
    functor = Functor.id(Diagram)

    def __call__(self, other, left=False):
        args = (other, self) if left else (self, other)
        return self.cod.application_factory(*args)

    def compose(self, *others: Term) -> biclosed.TermBase:
        """
        The composition of terms of function types: ``t.compose(u)`` for
        ``t : x >> y`` and ``u : y >> z`` is ``x(lambda v: u(t(v)))``, i.e.
        ``t`` is applied first, in the order of ``t >> u``, which composes
        the diagrams that terms also are and keeps its name.

        Parameters:
            others : Terms of function types, the exponent of each being
                the base of the previous one.

        Example
        -------
        >>> X, Y, Z = map(Ty, "XYZ")
        >>> t, u = (X >> Y)("t"), (Y >> Z)("u")
        >>> print(t.compose(u))
        X(lambda x: u(t(x)))
        """
        if not others:
            return self
        terms = (self, ) + others
        for term in terms:
            if not term.cod.is_exp:
                raise AxiomError(f"{term} is not of a function type.")
        for before, after in zip(terms, others):
            if before.cod.base != after.cod.exponent:
                raise AxiomError(messages.NOT_COMPOSABLE.format(
                    before, after, before.cod.base, after.cod.exponent))
        var = self.cod.variable_factory.fresh(
            "x", self.cod.exponent, *terms)
        body = reduce(lambda argument, func: func(argument), terms, var)
        return self.cod.abstraction_factory(var, body)

    def weaken(self, functor: Functor, context=None) -> Diagram:
        """
        The structural morphism from the image of a context to that of the
        free variables of the term: it discards the variables that do not
        occur in the term and permutes the others into the order of
        :attr:`freevars`.

        Parameters:
            functor : The functor to evaluate the types.
            context : A list of distinct variables containing the free ones,
                the free variables themselves by default.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> x, y = Variable("x", X), Variable("y", Y)
        >>> assert x.weaken(x.functor, [y, x])\\
        ...     == Diagram.swap(Y, X) >> X @ Diagram.discard(Y)
        """
        context = self.freevars if context is None else list(context)
        if context == self.freevars:
            return functor.cod.id(functor(self.dom))
        unused = [x for x in context if x not in self.freevars]
        permutation = functor.cod.permutation(
            [context.index(x) for x in self.freevars + unused],
            [functor(x.cod) for x in context])
        if not unused:
            return permutation  # ty: ignore[invalid-return-type]
        discard = functor.cod.discard(functor(
            self.ob().tensor(*[x.cod for x in unused])))
        return permutation >> functor.cod.id(functor(self.dom)) @ discard

    def share(self, functor: Functor,
              *contexts: list[biclosed.Variable]) -> Diagram:
        """
        The structural morphism from the image of the free variables of the
        term to that of the concatenation of ``contexts``, copying each
        variable once for each of the contexts it is in, i.e. how the
        subterms of a term share its free variables.

        Parameters:
            functor : The functor to evaluate the types.
            contexts : Lists of distinct free variables, one per subterm,
                each variable being in at least one of them.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> x, y = Variable("x", X), Variable("y", Y)
        >>> term = Tuple(x, y, x)
        >>> assert term.share(term.functor, [x], [y], [x])\\
        ...     == Diagram.copy(X) @ Y >> X @ Diagram.swap(X, Y)
        """
        counts = {x: sum(x in c for c in contexts) for x in self.freevars}
        source = [(x, i) for x in self.freevars for i in range(counts[x])]
        target = [(x, sum(x in c for c in contexts[:j]))
                  for j, context in enumerate(contexts) for x in context]
        copy = functor.cod.id(functor(self.ob())).tensor(*[
            functor.cod.copy(functor(x.cod), counts[x]) if counts[x] > 1
            else functor.cod.id(functor(x.cod)) for x in self.freevars])
        return copy >> functor.cod.permutation(
            [source.index(pair) for pair in target],
            [functor(x.cod) for x, _ in source])

    def eval_unpacked(self, functor=None, context=None):
        """
        The evaluation of a term followed by the unpacking of its product
        codomain, overridden by :class:`Tuple` so that binding a literal
        tuple never produces a :class:`Pack` followed by an
        :class:`Unpack`.
        """
        functor = functor or self.functor
        result = self.eval(
            functor, context)  # ty: ignore[too-many-positional-arguments]
        return result >> functor(Unpack(self.cod))\
            if self.cod.is_product else result

    def occurrences(self, variable: Variable) -> int:
        "The number of free occurrences of a variable in the term."
        # pylint: disable=unused-argument  # a constant has no variable
        return 0

    def substitute(self, substitution: Substitution) -> Self:
        "The term with the free variables of a substitution replaced."
        # pylint: disable=unused-argument  # a constant has no variable
        return self

    @classmethod
    def from_biclosed(cls, term: biclosed.TermBase) -> Term:
        """
        Translate a biclosed term into a closed term, dropping planarity by
        collapsing left and right exponentials and applications.

        Parameters:
            term : The biclosed term to translate.

        Note
        ----
        This method is inherited by :class:`Constant`, :class:`Variable`,
        :class:`Application` and :class:`Abstraction`, i.e. every closed
        :class:`Term`.

        Example
        -------
        >>> X, Y = biclosed.Ty("X"), biclosed.Ty("Y")
        >>> g, x = (Y << X)("g"), X("x")
        >>> print(TermBase.from_biclosed(g(x)))
        g(x)
        """
        functor = biclosed.Functor(
            ob_map=lambda x: cls.ob(x.inside[0].name),
            ar_map=lambda c: cls.ob.constant_factory(c.name, functor(c.cod)),
            dom=biclosed.Diagram, cod=cls.functor.cod)
        return functor(term)

    @classmethod
    def strategy(cls, **params):
        """
        Generate closed terms, which need not be linear, see
        :meth:`generate`.

        Parameters:
            params : Passed to :meth:`biclosed.TermBase.shapes`.
        """
        return super().strategy(**params)

    @classmethod
    def generate(cls, cod, choices, types, letters):
        """
        Build a term as :meth:`biclosed.TermBase.generate` does, except
        that a closed term need not be linear: its :class:`biclosed.Sampler`
        may use a bound variable any number of times, or not at all.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> choices = [4, 2, 0, 2, 0, 0, 2, 2]
        >>> print(TermBase.generate(Y << X, choices, [X], "x"))
        X(lambda x0: c0(x0)(x0))
        >>> print(TermBase.generate(Y << X, [4, 0], [X], "x"))
        X(lambda x0: c0)
        """
        return biclosed.Sampler(
            cls, iter(choices), types, letters, linear=False).term(cod)

    def reduce(self, budget: int | None = None,
               strategy: type[Strategy] | None = None) -> BohmTree | None:
        """
        The head normal form of the term as a :class:`BohmTree`, contracting
        at most ``budget`` beta redexes to reach it, or ``None`` when the
        budget runs out before a head normal form is reached. The tree's
        arguments are computed lazily, i.e. indexing into it may itself run
        out of budget, see :meth:`BohmTree.__getitem__`.

        Parameters:
            budget : The number of beta redexes to contract, no bound if
                ``None``.
            strategy : The subclass of :class:`Strategy` to follow,
                :class:`LeftmostOutermost` by default.

        Example
        -------
        >>> X = Ty('X')
        >>> f, x = Variable('f', X >> X), Variable('x', X)
        >>> two = (X >> X)(lambda f: X(lambda x: f(f(x))))
        >>> tree = two.reduce()
        >>> assert tree.spine == (f(x), )
        >>> assert tree[0][0] == BohmTree(X, (f, x), 1, tree.strategy, ())
        >>> assert two(X(lambda x: x))(x).reduce(budget=1) is None
        """
        return (strategy or self.strategy_factory)(budget)(
            self, tuple(self.freevars))  # ty: ignore[invalid-argument-type]

    def normal_form(self, budget: int | None = None,
                    strategy: type[Strategy] | None = None) -> Term:
        """
        The beta normal form of the term, i.e. the term of its Böhm tree,
        so that normalisation is idempotent.

        Parameters:
            budget : The number of beta redexes to contract, no bound if
                ``None``.
            strategy : The subclass of :class:`Strategy` to follow,
                :class:`LeftmostOutermost` by default.

        Example
        -------
        >>> X = Ty('X')
        >>> x = Variable('x', X)
        >>> term = X(lambda x: x)(x)
        >>> assert term.normal_form() == x == x.normal_form()
        >>> term.normal_form(budget=0)
        Traceback (most recent call last):
            ...
        ValueError: The budget of 0 beta redexes ran out reducing \
X(lambda x: x)(x).
        """
        tree = self.reduce(budget, strategy)
        if tree is None:
            raise ValueError(
                f"The budget of {budget} beta redexes ran out "
                f"reducing {self}.")
        return tree.to_term(len(self.freevars))


type Term = Constant | Variable | Application | Abstraction\
    | Tuple | Projection | Let


class Constant(TermBase, biclosed.Constant):
    """
    A constant term, evaluated in a context by discarding it. It prints as
    its bare name, so that terms read like textbook effectful lambda
    calculus and ``eval(str(term)) == term`` under the obvious variable
    naming convention, e.g. ``query = (E >> E)("query")``.
    """
    def __str__(self):
        return self.name

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        return self.weaken(functor, context)\
            >> biclosed.Constant.eval(self, functor)


class Variable(TermBase, biclosed.Variable):
    "A variable, evaluated in a context by discarding the other variables."
    def eval(self, functor=None, context=None):
        return self.weaken(functor or self.functor, context)

    def occurrences(self, variable):
        return int(self == variable)

    def substitute(self, substitution):
        return substitution.inside.get(self, self)


class Application(TermBase, biclosed.Application):
    """
    The application ``func(args)`` of a term to another.

    Attributes:
        overlap : The variables free in both ``func`` and ``args``, which
            :meth:`eval` copies.
    """
    func: Term
    args: Term

    def __init__(self, func: Term, args: Term, left: bool = False):
        # pylint: disable=unused-argument  # a closed category is symmetric
        biclosed.Application.__init__(self, func, args)

    def __check_dom__(self, func, args, left):
        self.overlap = [x for x in func.freevars if x in args.freevars]
        self.freevars = list(dict.fromkeys(func.freevars + args.freevars))
        return self.ob().tensor(*[x.cod for x in self.freevars])

    @property
    def is_linear(self):
        return not self.overlap\
            and self.func.is_linear and self.args.is_linear

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        func, args = self.func, self.args
        evaluate = functor.cod.ev(
            functor(func.cod.base), functor(func.cod.exponent))
        return self.weaken(functor, context)\
            >> self.share(functor, func.freevars, args.freevars)\
            >> func.eval(functor) @ args.eval(functor) >> evaluate

    def occurrences(self, variable):
        return self.func.occurrences(variable)\
            + self.args.occurrences(variable)

    def substitute(self, substitution):
        return type(self)(
            self.func.substitute(substitution),
            self.args.substitute(substitution))


class Abstraction(TermBase, biclosed.Abstraction):
    """
    The abstraction ``var.cod(lambda var: body)`` of a variable in a term,
    which need not occur in it or may occur several times.
    """
    var: Variable
    body: Term

    def __init__(self, var: Variable, body: Term, left: bool = False):
        # pylint: disable=unused-argument  # a closed category is symmetric
        biclosed.Abstraction.__init__(self, var, body)

    def __check_dom__(self):
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    @property
    def is_linear(self):
        return self.body.is_linear and self.body.occurrences(self.var) == 1

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        body = self.body.eval(functor, [self.var] + self.freevars)
        return self.weaken(functor, context)\
            >> body.curry(len(functor(self.var.cod)), left=False)

    def occurrences(self, variable):
        return 0 if variable == self.var else self.body.occurrences(variable)

    def substitute(self, substitution):
        inside = {key: value for key, value in substitution.inside.items()
                  if key != self.var and key in self.body.freevars}
        var, body = self.var, self.body
        if any(var in value.freevars for value in inside.values()):
            var = type(var).fresh(var.name, var.cod, body, *inside.values())
            body = Substitution({self.var: var})(body)
        return type(self)(var, Substitution(inside)(body))


class Compound(TermBase):
    """
    A term built from other terms, its :attr:`parts`: a :class:`Tuple`, a
    :class:`Projection` or a :class:`Let`. Two compound terms are equal when
    their parts are, and the name is printed from the parts when first read
    rather than when the term is built, so that building a term is linear in
    its size where printing every subterm made it quadratic.

    Parameters:
        dom : The domain of the term, i.e. the types of its free variables.
        cod : The codomain of the term.
    """
    def __init__(self, dom: biclosed.Ty, cod: biclosed.Ty):
        super().__init__(type(self).__name__, dom, cod)
        del self.name

    @property
    def parts(self) -> tuple:
        "The terms and other data the compound term is built from."
        raise NotImplementedError

    def setoid(self):
        return (type(self), ) + self.parts


class Tuple(Compound):
    """
    The tupling of terms, its codomain is the :class:`Product` of theirs.
    The empty tuple has the empty type as codomain, i.e. the nullary
    product is strict, so that terms of unit type stay type-preserving.

    Parameters:
        terms : The terms inside the tuple.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Tuple(x, y).cod == X * Y
    >>> assert Tuple(x, Tuple(y, x)).cod == X * (Y * X)
    >>> assert Tuple().cod == Ty()
    """
    def __init__(self, *terms: Term):
        for term in terms:
            assert_isinstance(term, TermBase)
        self.terms = terms
        self.freevars = list(dict.fromkeys(
            sum([term.freevars for term in terms], [])))
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        cod = self.ob(self.ob.product_factory(*[t.cod for t in terms]))\
            if terms else self.ob()
        super().__init__(dom, cod)

    @property
    def parts(self):
        return self.terms

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # Compound deletes it
        "The term as printed, computed once from its parts when first read."
        return f"Tuple({', '.join(map(str, self.terms))})"

    @property
    def is_linear(self):
        return all(term.is_linear for term in self.terms) and len(
            self.freevars) == sum(len(term.freevars) for term in self.terms)

    def eval(self, functor=None, context=None, pack=True):
        functor = functor or self.functor
        result = self.weaken(functor, context)\
            >> self.share(functor, *[term.freevars for term in self.terms])\
            >> functor.cod.id(functor(self.ob())).tensor(
                *[term.eval(functor) for term in self.terms])
        return result >> functor(Pack(self.cod))\
            if pack and self.terms else result

    def eval_unpacked(self, functor=None, context=None):
        return self.eval(functor, context, pack=False)

    def occurrences(self, variable):
        return sum(term.occurrences(variable) for term in self.terms)

    def substitute(self, substitution):
        return type(self)(*[
            term.substitute(substitution) for term in self.terms])

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Tuples of one length with alpha-equivalent terms. """
        if any(type(other) is not type(self)
               or len(other.terms) != len(self.terms) for other in others):
            return False
        return all(term.alpha_eq_under(
            scopes, [other.terms[i] for other in others], depth)
            for i, term in enumerate(self.terms))

    def map(self, functor, context):
        return type(self)(*[
            functor.map_term(term, context) for term in self.terms])

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    @property
    def constants(self):
        "The constants of the term, in order of occurrence."
        return sum([term.constants for term in self.terms], [])

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'terms': [term.to_tree() for term in self.terms]}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, tree['terms']))


class Projection(Compound):
    """
    The projection onto one factor of a term with a :class:`Product` type,
    which evaluation interprets by discarding the other factors.

    Parameters:
        arg : The term to project from, with a product type as codomain.
        index : The index of the factor to project onto.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Projection(Tuple(x, y), 1).cod == Y
    """
    def __init__(self, arg: Term, index: int):
        assert_isinstance(arg, TermBase)
        assert_isinstance(index, int)
        if not arg.cod.is_product:
            raise TypeError(f"Expected {Product}, got {arg.cod!r}")
        if not 0 <= index < len(arg.cod.factors):
            raise IndexError(f"{arg.cod!r} has no factor {index}")
        self.arg, self.index = arg, index
        self.freevars = arg.freevars
        super().__init__(arg.dom, arg.cod.factors[index])

    @property
    def parts(self):
        return self.arg, self.index

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # Compound deletes it
        "The term as printed, computed once from its parts when first read."
        return f"Projection({self.arg}, {self.index})"

    @property
    def is_linear(self):
        return self.arg.is_linear and len(self.arg.cod.factors) == 1

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        discards = functor.cod.id(functor(self.ob())).tensor(*[
            functor.cod.id(functor(typ)) if i == self.index
            else functor.cod.discard(functor(typ))
            for i, typ in enumerate(self.arg.cod.factors)])
        return self.arg.eval_unpacked(functor, context) >> discards

    def occurrences(self, variable):
        return self.arg.occurrences(variable)

    def substitute(self, substitution):
        return type(self)(self.arg.substitute(substitution), self.index)

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Projections onto one index of alpha-equivalent terms. """
        if any(type(other) is not type(self) or other.index != self.index
               for other in others):
            return False
        return self.arg.alpha_eq_under(
            scopes, [other.arg for other in others], depth)

    def map(self, functor, context):
        return type(self)(functor.map_term(self.arg, context), self.index)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg!r}, {self.index!r})"

    @property
    def constants(self):
        "The constants of the term, in order of occurrence."
        return self.arg.constants

    def to_tree(self):
        return {'factory': factory_name(type(self)),
                'arg': self.arg.to_tree(), 'index': self.index}

    @classmethod
    def from_tree(cls, tree):
        return cls(from_tree(tree['arg']), tree['index'])


class Let(Compound):
    """
    The evaluation of an ``expression`` term, binding a tuple of
    ``variables`` to the factors of its result inside a ``body`` term,
    i.e. the statement ``let (x, ..., z) = expression in body``.

    Parameters:
        expression : The term that is evaluated.
        variables : The variables binding the factors of the result.
        body : The term in which the variables are bound.

    Note
    ----
    The codomain of ``expression`` unpacks either as the factors of its
    :class:`Product` type or as the tensor of the variables' types. Bound
    variables may be discarded or copied by the body, see :func:`let` for
    the introspection helper that builds the statement from a function.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x, y = (X >> Y)("f"), Variable("x", X), Variable("y", Y)
    >>> term = Let(f(x), (y, ), Tuple(y, y))
    >>> print(term)
    let(f(x), lambda y: Tuple(y, y))
    >>> assert term.cod == Y * Y and not term.is_linear
    """
    def __init__(self, expression: Term, variables: tuple[Variable, ...],
                 body: Term):
        assert_isinstance(expression, TermBase)
        assert_isinstance(body, TermBase)
        variables = tuple(variables)
        for var in variables:
            assert_isinstance(var, Variable)
        if len(set(variables)) != len(variables):
            raise ValueError(f"Expected distinct variables, got {variables}")
        if set(variables).intersection(expression.freevars):
            raise ValueError(f"{variables} are free in {expression}")
        cods = [x.cod for x in variables]
        matched = list(expression.cod.factors) == cods\
            if expression.cod.is_product\
            else self.ob().tensor(*cods) == expression.cod
        if not matched:
            raise ValueError(
                f"Expected variables of type {expression.cod}, got {cods}")
        self.expression, self.variables, self.body\
            = expression, variables, body
        self.freevars = list(dict.fromkeys(expression.freevars + [
            x for x in body.freevars if x not in variables]))
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        super().__init__(dom, body.cod)

    @property
    def parts(self):
        return self.expression, self.variables, self.body

    @cached_property
    def name(self):  # pylint: disable=method-hidden  # Compound deletes it
        "The term as printed, computed once from its parts when first read."
        params = " " + ", ".join(x.name for x in self.variables)\
            if self.variables else ""
        return f"let({self.expression}, lambda{params}: {self.body})"

    @property
    def rest(self) -> list[biclosed.Variable]:
        "The free variables of the body that the statement does not bind."
        return [x for x in self.freevars if x in self.body.freevars]

    @property
    def is_linear(self):
        return self.expression.is_linear and self.body.is_linear\
            and not set(self.expression.freevars).intersection(self.rest)\
            and all(self.body.occurrences(x) == 1 for x in self.variables)

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        rest = functor.cod.id(functor(
            self.ob().tensor(*[x.cod for x in self.rest])))
        return self.weaken(functor, context)\
            >> self.share(functor, self.expression.freevars, self.rest)\
            >> self.expression.eval_unpacked(functor) @ rest\
            >> self.body.eval(functor, list(self.variables) + self.rest)

    def occurrences(self, variable):
        return self.expression.occurrences(variable) + (
            0 if variable in self.variables
            else self.body.occurrences(variable))

    def substitute(self, substitution):
        inside = {key: value for key, value in substitution.inside.items()
                  if key not in self.variables and key in self.body.freevars}
        expression = self.expression.substitute(substitution)
        variables, body = list(self.variables), self.body
        for i, var in enumerate(variables):
            if any(var in value.freevars for value in inside.values()):
                variables[i] = type(var).fresh(
                    var.name, var.cod, expression, body,
                    *variables, *inside.values())
                body = Substitution({var: variables[i]})(body)
        return type(self)(
            expression, tuple(variables), Substitution(inside)(body))

    def alpha_eq_under(self, scopes, others, depth=0):
        """
        Statements binding variables of the same types to alpha-equivalent
        expressions, with bodies alike under the levels they bind.
        """
        if any(type(other) is not type(self)
               or [x.cod for x in other.variables]
               != [x.cod for x in self.variables] for other in others):
            return False
        terms = (self, *others)
        return self.expression.alpha_eq_under(
            scopes, [other.expression for other in others], depth)\
            and self.alpha_eq_bound(
                scopes, [term.body for term in terms],
                [term.variables for term in terms], depth)

    def map(self, functor, context):
        expression = functor.map_term(self.expression, context)
        context = {key: value for key, value in context.items()
                   if key not in self.variables}
        variables = []
        for var in self.variables:
            context[var] = functor.map_variable(var, context)
            variables.append(context[var])
        return type(self)(expression, tuple(variables),
                          functor.map_term(self.body, context))

    def __repr__(self):
        return factory_name(type(self)) + f"({self.expression!r}, "\
            + f"{self.variables!r}, {self.body!r})"

    @property
    def constants(self):
        "The constants of the term, in order of occurrence."
        return self.expression.constants + self.body.constants

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'expression': self.expression.to_tree(),
            'variables': [var.to_tree() for var in self.variables],
            'body': self.body.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(
            from_tree(tree['expression']),
            tuple(map(from_tree, tree['variables'])),
            from_tree(tree['body']))


def let(expression: Term, body: Callable) -> Let:
    """
    Bind the result of an ``expression`` term inside the ``body`` of a
    Python function, whose variable names are given by introspection and
    whose types are the factors of the codomain of ``expression``.

    Parameters:
        expression : The term that is evaluated.
        body : A function from the bound variables to a term.

    Example
    -------
    The term for the self-attention block of the CatGPT benchmark, where
    an embedded token is packed into a query, key and value before
    attention and a feed-forward layer are applied:

    >>> E = Ty("E")
    >>> query, key, value = [(E >> E)(name) for name in (
    ...     "query", "key", "value")]
    >>> attention = (E.product(E, E) >> E)("attention")
    >>> feed_forward = (E >> E)("feed_forward")
    >>> block = E(lambda x: let(Tuple(query(x), key(x), value(x)),
    ...     lambda q, k, v: let(attention(Tuple(q, k, v)),
    ...         lambda a: feed_forward(a))))
    >>> assert block.cod == E >> E
    >>> block.draw(doctest="docs/_static/closed/catgpt-block.svg",
    ...     aspect="auto", figsize=(6, 8), margins=(0.2, 0))

    .. image:: /_static/closed/catgpt-block.svg
        :align: center
    """
    parameters = signature(body).parameters.values()
    if any(x.kind not in (x.POSITIONAL_ONLY, x.POSITIONAL_OR_KEYWORD)
           for x in parameters):
        raise ValueError(
            f"Expected positional parameters, got {signature(body)}")
    varnames = [x.name for x in parameters]
    cod = expression.cod
    factors = list(cod.factors) if cod.is_product\
        else [cod[i:i + 1] for i in range(len(cod))]
    if len(varnames) != len(factors):
        raise ValueError(
            f"Expected {len(factors)} variables, got {len(varnames)}")
    variables = tuple(typ.variable_factory(name, typ)
                      for name, typ in zip(varnames, factors))
    return Let(expression, variables, body(*variables))


@dataclass
class Substitution:
    """
    The simultaneous, capture-avoiding substitution of terms for variables.

    Parameters:
        inside : The term substituted for each variable, of the same type.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x, y = (X >> Y)("f"), Variable("x", X), Variable("y", X)
    >>> assert Substitution({x: y})(X(lambda y: f(x))) == X(lambda y_: f(y))
    """
    inside: Dict[Variable, Term]

    def __post_init__(self):
        for variable, term in self.inside.items():
            if not isinstance(variable, Variable)\
                    or not isinstance(term, TermBase):
                raise TypeError
            if variable.cod != term.cod:
                raise ValueError(
                    f"Expected {variable.cod}, got {term.cod}")

    def __call__(self, term: Term) -> Term:
        return term.substitute(self)


@dataclass
class BohmTree:
    """
    A `Böhm tree <https://en.wikipedia.org/wiki/B%C3%B6hm_tree>`_ is the
    normal form of a :class:`Term`: each node abstracts the variables in
    scope beyond those of its parent, then applies a head variable to the
    trees of its arguments. Arguments are computed lazily and cached, i.e.
    ``tree[i]`` reduces ``spine[i]`` with ``strategy`` on first access and
    raises :class:`ValueError` if its budget has run out.

    Parameters:
        head_cod : The type of the head variable applied to the arguments,
            i.e. of the tree without abstracting its own bound variables.
            Use :meth:`cod` for the type of the term it represents.
        variables : The variables in scope, those of the parent followed by
            the ones bound at this node.
        head : The index in ``variables`` of the head variable.
        strategy : The strategy used to reduce the arguments, shared with
            every node of the tree so that its budget is spent only once.
        spine : The unreduced terms of the arguments of the head variable,
            in their original left-to-right order.

    Example
    -------
    >>> from discopy import cat, closed
    >>> X = Ty('X')
    >>> f, x = Variable('f', X >> X), Variable('x', X)
    >>> tree = (X >> X)(lambda f: X(lambda x: f(x))).reduce()
    >>> assert tree.spine == (x, )
    >>> assert tree[0] == BohmTree(X, (f, x), 1, tree.strategy, ())
    >>> assert tree == eval(repr(tree))
    >>> print(tree.to_term())
    (X >> X)(lambda f: X(lambda x: f(x)))
    """
    head_cod: Ty
    variables: tuple[Variable, ...]
    head: int
    strategy: Strategy = field(compare=False)
    spine: tuple[Term, ...]
    cache: dict[int, BohmTree] = field(
        default_factory=dict, compare=False, repr=False, init=False)

    def __post_init__(self):
        assert_isinstance(self.head_cod, Ty)
        assert_isinstance(self.head, int)
        assert_isinstance(self.strategy, Strategy)
        if not 0 <= self.head < len(self.variables):
            raise AxiomError(
                f"Expected 0 <= head < {len(self.variables)}, "
                f"got {self.head} instead.")
        head_cod = self.variables[self.head].cod
        for term in self.spine:
            assert_isinstance(term, TermBase)
            if not head_cod.is_exp:
                raise AxiomError(
                    f"Expected an exponential type, got {head_cod}.")
            if term.cod != head_cod.exponent:
                raise AxiomError(
                    f"Expected argument.cod == {head_cod.exponent}, "
                    f"got {term.cod} instead.")
            head_cod = head_cod.base
        if head_cod != self.head_cod:
            raise AxiomError(
                f"Expected result type {head_cod}, "
                f"got {self.head_cod} instead.")

    def cod(self, n: int = 0) -> Ty:
        """
        The type of the term of the tree, abstracting over ``variables[n:]``
        for ``n`` the number of variables of the parent.

        Parameters:
            n : The number of variables bound by the parent.
        """
        result = self.head_cod
        for var in reversed(self.variables[n:]):
            result = var.cod >> result
        return result

    def __getitem__(self, key: int) -> BohmTree:
        """
        The tree of the ``key``-th argument, reduced from ``spine[key]`` on
        first access and cached thereafter.

        Parameters:
            key : The index of the argument in :attr:`spine`.
        """
        if key not in self.cache:
            tree = self.strategy(self.spine[key], self.variables)
            if tree is None:
                raise ValueError(f"Missing arguments in {self}.")
            self.cache[key] = tree
        return self.cache[key]

    def to_term(self, n: int = 0) -> Term:
        """
        The term of the tree in the standard syntax, abstracting over
        ``variables[n:]`` for ``n`` the number of variables of the parent.
        Forces every argument, in the order chosen by ``strategy``, raising
        :class:`ValueError` if any of them is missing.

        Parameters:
            n : The number of variables bound by the parent.
        """
        for i in self.strategy.order(self.spine):
            self[i]
        result = self.variables[self.head]
        for i in range(len(self.spine)):
            result = result(self[i].to_term(len(self.variables)))
        for var in reversed(self.variables[n:]):
            result = var.cod.abstraction_factory(var, result)
        return result  # ty: ignore[invalid-return-type]

    def __repr__(self):
        return factory_name(type(self)) + (
            f"({self.head_cod!r}, {self.variables!r}, {self.head}, "
            f"{self.strategy!r}, {self.spine!r})")


@dataclass
class Strategy:
    """
    A reduction strategy computes the :class:`BohmTree` of a term within a
    ``budget`` of beta redexes to contract, with no bound when ``None``.

    Contracting the head redex comes first in every strategy that reaches a
    head normal form; the arguments of the head variable are left lazy in
    the resulting tree, see :meth:`BohmTree.__getitem__`. Concrete
    strategies such as :class:`LeftmostOutermost` choose the order in which
    the arguments are forced, e.g. by :meth:`BohmTree.to_term`, by
    implementing :meth:`Strategy.order`.

    Parameters:
        budget : The number of beta redexes left to contract.
    """
    budget: int | None = None

    def __call__(self, term: Term, variables: tuple[Variable, ...] = ()
                 ) -> BohmTree | None:
        """
        The head normal form of a term in a scope of variables as a
        :class:`BohmTree` with its arguments left lazy, or ``None`` when the
        budget runs out before a head normal form is reached.

        Parameters:
            term : The term to reduce.
            variables : The variables in scope, i.e. the free variables of
                the term followed by the ones bound by its parents.
        """
        scope: list[Variable] = list(variables)
        spine = []
        while isinstance(term, (Application, Abstraction)):
            if isinstance(term, Application):
                spine.append(term.args)
                term = term.func
            elif spine:
                if not self.spend():
                    return None
                term = Substitution({term.var: spine.pop()})(term.body)
            else:
                scope.append(term.var)
                term = term.body
        assert_isinstance(term, Variable)
        index = scope[::-1].index(term)  # ty: ignore[invalid-argument-type]
        head = len(scope) - 1 - index
        cod = term.cod
        for _ in spine:
            cod = cod.base
        return BohmTree(
            cod,  # ty: ignore[invalid-argument-type]
            tuple(scope), head, self, tuple(reversed(spine)))

    def spend(self) -> bool:
        "Take one beta redex from the budget, or return ``False``."
        if self.budget is not None and self.budget <= 0:
            return False
        self.budget = None if self.budget is None else self.budget - 1
        return True

    def order(self, terms: tuple[Term, ...]) -> tuple[int, ...]:
        """
        The order, as indices into ``terms``, in which to force the
        arguments of a head variable when forcing the whole tree, e.g. in
        :meth:`BohmTree.to_term`.

        Parameters:
            terms : The unreduced arguments of the head variable, i.e. a
                :class:`BohmTree`'s :attr:`BohmTree.spine`.
        """
        raise NotImplementedError


class LeftmostOutermost(Strategy):
    """
    The default :class:`Strategy`: contract the head redex, then force the
    arguments of the head variable from left to right.

    Example
    -------
    >>> X = Ty('X')
    >>> f, x = Variable('f', X >> X), Variable('x', X)
    >>> term = (X >> X)(lambda f: X(lambda x: f(X(lambda x: x)(x))))
    >>> tree = LeftmostOutermost(0)(term)
    >>> try:
    ...     tree[0]
    ... except ValueError:
    ...     print("starved")
    starved
    >>> tree = LeftmostOutermost(1)(term)
    >>> assert tree == term.reduce(budget=1)
    >>> assert tree[0] == BohmTree(X, (f, x), 1, tree.strategy, ())
    """
    def order(self, terms):
        return tuple(range(len(terms)))


class RightmostFirst(LeftmostOutermost):
    """
    A :class:`Strategy` that contracts the head redex, then forces the
    arguments of the head variable from right to left, e.g. to reduce the
    last argument first within a fixed budget.

    Example
    -------
    >>> X = Ty('X')
    >>> c, d = Variable('c', X), Variable('d', X)
    >>> h = Variable('h', X >> (X >> X))
    >>> term = h(X(lambda x: x)(c))(X(lambda x: x)(d))
    >>> tree = term.reduce(budget=1, strategy=RightmostFirst)
    >>> assert tree.strategy.order(tree.spine) == (1, 0)
    >>> assert tree[1].to_term(len(tree.variables)) == d
    """
    def order(self, terms):
        return tuple(reversed(range(len(terms))))


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Ty.product_factory = Product
TermBase.strategy_factory = LeftmostOutermost

for law in sorted(set(Box.axioms) - set(Serialisable.axioms)):
    setattr(TermBase, law, getattr(Box, law).inapplicable(
        "A term is a box: the laws of its category are the diagram's."))


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
