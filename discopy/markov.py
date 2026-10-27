# -*- coding: utf-8 -*-

"""
The free Markov category, i.e. a semicartesian category with a supply of
commutative comonoid, see :cite:t:`FritzLiang23`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Function
    Swap
    Permutation
    Copy
    Discard
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor


Axioms
------

>>> x = Ty('x')

>>> copy, delete = Copy(x), Copy(x, n=0)

Cocommutative comonoid
======================

>>> counitality = Equation(copy >> delete @ x, Id(x), copy >> x @ delete)
>>> coassociativity = Equation(copy >> copy @ x, copy >> x @ copy)
>>> cocommutativity = Equation(copy >> Swap(x, x), copy)
>>> assert counitality and coassociativity and cocommutativity
>>> Equation(counitality, coassociativity, cocommutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/comonoid.svg")

.. image:: /_static/frobenius/comonoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.copy(x @ x, n=0), delete @ delete)
>>> assert Equation(Diagram.copy(x @ x),
...     copy @ copy >> x @ Swap(x, x) @ x)

Note
----
Equality of Markov diagrams is computed by translation to hypergraph.
Copy boxes are translated to spiders, thus when they appear in the same
diagram as the merge boxes of :mod:`comarkov` they automatically satisfy
the :mod:`frobenius` axioms.
"""

from __future__ import annotations

import warnings
from collections.abc import Sized
from operator import index
from typing import ClassVar

from discopy import symmetric, monoidal, feedback, cmap, hypergraph, messages
from discopy.abc import ComarkovCategory, MarkovCategory, Nat
from discopy.axioms import FUEL
from discopy.cat import factory
from discopy.feedback import (  # noqa: F401  pylint: disable=unused-import
    Wire, Ty)
from discopy.python import finset
from discopy.utils import (
    AxiomError, assert_isatomic, assert_iscomposable, deprecated_alias,
    factory_name, from_tree)


class Layer(symmetric.Layer):
    """
    A tensor product of generators and non-empty plumbing, where plumbing is
    a type when it is the identity and a :class:`Function` otherwise, i.e.
    copies, discards and permutations.

    Plumbing components are coalesced, so that the layer alternates between
    functions and generators, which can be consecutive.

    Parameters:
        inside : Generators and plumbing, with at least one generator or one
                 function other than the identity.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> assert Layer(Copy(x), y, f, Discard(y)).boxes_or_types == (
    ...     Function(x @ y, [0, 0, 1]), f, Discard(y))
    """


@factory
class Diagram(symmetric.Diagram, MarkovCategory):
    """
    A Markov diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    We can create arbitrary Markov diagrams with the standard notation for
    Python functions.

    >>> x = Ty('x')
    >>> f = Box('f', x, x)

    >>> copy_then_apply = Diagram.from_callable(x, x @ x)(
    ...     lambda x: (f(x), f(x)))

    >>> @Diagram.from_callable(x, x @ x)
    ... def apply_then_copy(x):
    ...     y = f(x)
    ...     return y, y

    >>> Equation(copy_then_apply, apply_then_copy, symbol="$\\\\neq$").draw(
    ...     doctest="docs/_static/markov/copy_and_apply.svg")

    .. image:: /_static/markov/copy_and_apply.svg

    Note
    ----
    A Markov diagram is a feedback diagram whose wires carry a time step, so
    that it is the syntax of streams, see :mod:`discopy.stream`.

    >>> x = Ty('x')
    >>> zero = Box('0', Ty(), x.head)
    >>> rand = Box('rand', Ty(), x)
    >>> plus = Box('+', x @ x, x)
    >>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
    ...         >> FollowedBy(x) >> Copy(x)).feedback()
    >>> walk.draw(doctest="docs/_static/feedback/feedback-random-walk.svg")

    .. image:: /_static/feedback/feedback-random-walk.svg
        :align: center
    """
    copy_factory: ClassVar[type["Copy"]]
    discard_factory: ClassVar[type["Discard"]]
    function_factory: ClassVar[type["Function"]]
    layer_factory = Layer

    @classmethod
    def quotient_strategy(cls, *, dom=None, cod=None, fuel=FUEL):
        """
        Sample Markov diagrams through their quotient: a hypergraph drawn
        by :meth:`discopy.hypergraph.Hypergraph.strategy` with at most
        ``fuel + 1`` boxes, besides the one balancing box that may close its
        boundary, and the spiders that the category supports, i.e.
        copies and discards here and every spider in a hypergraph category,
        downgraded with :meth:`discopy.hypergraph.Hypergraph.to_diagram`.
        """
        category = hypergraph.Hypergraph[cls]  # ty: ignore[invalid-type-form]
        return category.strategy(
            dom=dom, cod=cod, max_leaves=fuel + 1).map(
                lambda graph: graph.to_diagram())

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        """
        The spiders of a Markov category: the copies, with one leg in, and
        the traces of copies, with none, i.e. what tracing a copy back into
        its own input encodes to as a hypergraph.

        >>> x = Ty('x')
        >>> assert Diagram.spider_factory(0, 1, x) == Copy(x).trace(left=True)
        """
        if phase is not None or n_legs_in not in (0, 1):
            raise ValueError(messages.NO_SUCH_SPIDER.format(
                factory_name(cls), "copies"))
        if not n_legs_in:
            return cls.copy(typ, n_legs_out + 1).trace(left=True)
        return cls.copy_factory(typ, n_legs_out)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.

        Note
        ----
        A category with native functions copies a type as one
        :class:`Function`.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.copy(x @ y) == Function(x @ y, [0, 1, 0, 1])
        """
        if cls.function_factory.ar is cls:
            return cls.from_function(finset.Function.copy(len(x), n), x)
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(
            cls, 1, n, x)  # ty: ignore[invalid-argument-type]

    @classmethod
    def discard(cls, x: monoidal.Ty) -> Diagram:
        """
        The discard of a type :code:`x`, i.e. no copy of it at all.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)

    map_agreement = monoidal.Diagram.map_agreement.failing(
        messages.MAP_HAS_NO_SPIDERS)
    str_transparency = monoidal.Diagram.str_transparency.failing(
        messages.STR_SPLITS_LAYERS)

    @classmethod
    def from_function(cls, func, dom: monoidal.Ty) -> Diagram:
        """
        A native :class:`Function` box when the category has a matching
        function factory, its decomposition by
        :meth:`discopy.abc.MarkovCategory.function` otherwise. A bijection
        becomes :meth:`from_permutation`, the identity the identity diagram.

        Parameters:
            func : The input each output reads, as a list or a
                   :class:`finset.Function`.
            dom : The type to copy, discard and permute.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert Diagram.from_function([1, 0, 0], x @ y)\\
        ...     == Function(x @ y, [1, 0, 0])
        >>> assert Diagram.from_function([0, 1], x @ y) == Id(x @ y)
        """
        func = list(func)
        if sorted(func) == list(range(len(dom))):
            return cls.from_permutation(
                func, dom)  # ty: ignore[invalid-return-type]
        if cls.function_factory.ar is cls:
            return cls.function_factory(dom, func)
        return super().from_function(func, dom)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Deprecated: a Markov diagram has no merge, see
        :meth:`discopy.comarkov.Diagram.merge`, which a diagram that is
        also comarkov, e.g. :class:`discopy.frobenius.Diagram`, inherits.

        Parameters:
            x : The type to merge.
            n : The number of copies.
        """
        if issubclass(cls, ComarkovCategory):
            return super().merge(x, n)
        warnings.warn(
            f"{factory_name(cls)}.merge is deprecated, "
            "use discopy.comarkov.Diagram.merge instead.",
            DeprecationWarning, stacklevel=2)
        from discopy import comarkov
        return comarkov.Diagram.merge(x, n)  # ty: ignore[invalid-return-type]

    @property
    def is_deterministic(self) -> bool:
        """
        Whether copying commutes with the diagram, i.e. its boxes are
        deterministic, see :class:`Box`.

        >>> x = Ty('x')
        >>> sample = Box('sample', x, x, is_deterministic=False)
        >>> assert Copy(x).is_deterministic
        >>> assert not (Copy(x) >> sample @ x).is_deterministic
        """
        return all(box.is_deterministic for box in self.boxes)

    @property
    def is_affine(self) -> bool:
        """
        Whether discarding commutes with the diagram, i.e. its boxes are
        affine, see :class:`Box`.

        >>> x = Ty('x')
        >>> condition = Box('condition', x, Ty(), is_affine=False)
        >>> assert Copy(x).is_affine
        >>> assert not (Copy(x) >> condition @ x).is_affine
        """
        return all(box.is_affine for box in self.boxes)

    @property
    def is_cartesian(self) -> bool:
        """
        Whether the diagram is deterministic and affine, i.e. a total
        function: copying and discarding both commute with it.
        """
        return self.is_deterministic and self.is_affine


class Box(symmetric.Box, Diagram):
    """
    A Markov box is a symmetric box in a Markov diagram, a total function
    unless it says otherwise.

    A box ``f`` is deterministic when ``f >> copy == copy >> f @ f`` and
    affine when ``f >> discard == discard``, e.g. a sample is affine but not
    deterministic, a condition deterministic but not affine. A deterministic
    box that is not affine is a hard constraint, i.e. a partial function: a
    weight ``w`` copied is ``w * w``, so it is zero or one, and a soft score
    is not deterministic. Both flags are attributes of each box rather than
    of its class, as an identity or a swap is a total function whatever its
    class, and they are part of its equality. They are declared of the box
    as ``deterministic`` and ``affine`` and hold of it but not of its dagger,
    as the converse of a total function need not be one, so that the dagger
    of a dagger is the box again.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        is_deterministic : Whether copying commutes with the box.
        is_affine : Whether discarding commutes with the box.

    Example
    -------
    >>> p, x = Ty('p'), Ty('x')
    >>> sample = Box('sample', p, x, is_deterministic=False)
    >>> assert sample != Box('sample', p, x)
    >>> sample
    markov.Box('sample', feedback.Ty(feedback.Wire('p')), \
feedback.Ty(feedback.Wire('x')), is_deterministic=False)
    >>> assert not sample.dagger().dagger().is_deterministic

    >>> f = Box('f', p, x)
    >>> assert f.is_cartesian and not f.dagger().is_cartesian
    >>> assert f.dagger().dagger() == f and f.dagger().deterministic
    """
    serialised_attrs = symmetric.Box.serialised_attrs + (
        "is_deterministic", "is_affine")
    deterministic = affine = True

    def __init__(self, *args, is_deterministic=True, is_affine=True,
                 **params):
        self.deterministic, self.affine = is_deterministic, is_affine
        super().__init__(*args, **params)

    @property
    def is_deterministic(self) -> bool:
        """
        Whether the box is declared deterministic and is no dagger: setting it
        declares it, as the constructor does.
        """
        return self.deterministic and not self.is_dagger

    @is_deterministic.setter
    def is_deterministic(self, value: bool):
        self.deterministic = value

    @property
    def is_affine(self) -> bool:
        """
        Whether the box is declared affine and is no dagger: setting it
        declares it, as the constructor does.
        """
        return self.affine and not self.is_dagger

    @is_affine.setter
    def is_affine(self, value: bool):
        self.affine = value

    def flags(self) -> dict[str, bool]:
        """
        The flags declared of the box that differ from those of a total
        function, by name, as :meth:`__repr__` and :meth:`to_tree` write them
        down.
        """
        return {name: False for name, value in [
            ("is_deterministic", self.deterministic),
            ("is_affine", self.affine)] if not value}

    def __repr__(self):
        flags = "".join(f", {name}=False" for name in self.flags())
        if self.is_dagger or not flags:
            return super().__repr__()
        return super().__repr__()[:-1] + flags + ")"

    def setoid(self):
        """
        Markov boxes are equal when they are equal as :class:`cat.Box` and
        their flags are equal too.
        """
        return super().setoid() + (self.deterministic, self.affine)

    def flagged(self, other: Box) -> Box:
        """
        Give a box built from this one the flags declared of this one, as the
        rotations of a :class:`discopy.frobenius.Box` do.
        """
        other.deterministic, other.affine = self.deterministic, self.affine
        return other

    def rebuild(self, **changes) -> Box:
        """
        Rebuild a box with the flags declared of it, when its constructor
        takes them, else with those it computes, e.g. from its terms.

        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> x = Ty('x')
        >>> sample = Box('sample', x, x, data=phi, is_deterministic=False)
        >>> assert not sample.subs(phi, 1).is_deterministic
        >>> assert not sample.lambdify(phi)(1).is_deterministic
        >>> assert sample.rebuild(is_deterministic=True).is_deterministic
        >>> f = Box('f', x, x)
        >>> assert not Sum((f, )).rebuild(terms=(f, f)).is_deterministic
        """
        declared = {"is_deterministic": self.deterministic,
                    "is_affine": self.affine}
        return super().rebuild(**{
            name: value for name, value in declared.items()
            if name in self.serialised_attrs} | changes)

    def to_tree(self) -> dict:
        """
        Serialise a box, with the flags it does not have.

        >>> from discopy.utils import dumps, loads
        >>> box = Box('condition', Ty('x'), Ty(), is_affine=False)
        >>> assert loads(dumps(box)) == box
        """
        return {**super().to_tree(), **self.flags()}

    @classmethod
    def from_tree(cls, tree: dict):
        result = super().from_tree(tree)
        for name in ("is_deterministic", "is_affine"):
            if name in tree:
                setattr(result, name.removeprefix("is_"), tree[name])
        return result


class Function(Box):
    """
    A function between finite sets read backwards as plumbing: output ``i``
    is a copy of input ``func[i]``, i.e. ``cod[i] == dom[func[i]]``. An input
    that no output reads is discarded, one that several outputs read is
    copied, so that a function holds a :class:`finset.Function
    <discopy.python.finset.Function>`, an arrow of the free Markov category
    on one object.

    A :class:`Layer` stores it as plumbing rather than as a generator. A
    bijection constructs a :class:`Permutation`, a function on one wire a
    :class:`Copy` or a :class:`Discard`, and the identity function is the
    identity diagram, see :meth:`Diagram.from_function`.

    Parameters:
        dom : The domain, i.e. the wires to copy, discard and permute.
        func : The input each output reads, as a list or a
               :class:`finset.Function <discopy.python.finset.Function>`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Function(x @ y, [1, 0, 0])
    >>> assert f.cod == y @ x @ x
    >>> assert Equation(f, Swap(x, y) >> y @ Copy(x))
    >>> assert Function(x, [0, 0]) == Copy(x) and Function(x, []) == Discard(x)
    >>> assert Function(x @ y, [1, 0]) == Swap(x, y)

    Tensoring functions gives one function, as tensoring permutations does:

    >>> assert Copy(x) @ Discard(y) == Function(x @ y, [0, 0])
    >>> assert x @ Copy(y) == Function(x @ y, [0, 1, 1])

    A function is drawn as the copies of each input, then the permutation of
    the copies.

    >>> f.draw(doctest='docs/_static/markov/function.svg')

    .. image:: /_static/markov/function.svg
        :align: center
    """
    serialised_attrs = ("dom", "func")

    def __new__(cls, dom: monoidal.Ty | None = None, func=None):
        if dom is None or cls is not cls.ar.function_factory:
            return super().__new__(cls)
        func = list(func)  # ty: ignore[invalid-argument-type]
        if sorted(func) == list(range(len(dom))):
            return cls.ar.permutation_factory(dom, func)
        if len(dom) == 1 and not any(func):
            return cls.ar.copy_factory(dom, len(func))
        return super().__new__(cls)

    def __init__(self, dom: monoidal.Ty, func):
        self.func = finset.Function(list(func), Nat(len(dom)), Nat(len(func)))
        cod = dom[:0].tensor(*(dom[i] for i in self.func))
        Box.__init__(self, f"Function({list(self.func)})", dom, cod)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle and a delay read.

        >>> x, y = map(Ty, "xy")
        >>> f = Function(x @ y, [1, 0, 0])
        >>> assert f.delay() == Function((x @ y).delay(), [1, 0, 0])
        """
        return self.dom, list(self.func)

    @property
    def is_identity(self) -> bool:
        """ Whether the underlying function is the identity. """
        return list(self.func) == list(range(len(self.dom)))

    @property
    def size(self) -> int:
        """
        The number of copies and discards the function fuses, i.e. of its
        inputs not read exactly once, so that a permutation counts zero.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Copy(x).size == Discard(x).size == 1
        >>> assert Function(x @ y, [0, 0]).size == 2
        >>> assert Function(x @ y, [1, 0, 0]).size == 1
        """
        counts, _ = self.func.factorise()
        return sum(n != 1 for n in counts)

    def setoid(self):
        if self.is_identity:
            return (), self.dom, self.cod
        return type(self), self.dom, tuple(self.func)

    def to_drawing(self):
        """ Draw the copies of each input, then the permutation. """
        from discopy.drawing import Drawing
        counts, perm = self.func.factorise()
        copies = Drawing.id().tensor(*(
            self.copy_factory(self.dom[j:j + 1], n).to_drawing() if n != 1
            else Drawing.id(self.dom[j:j + 1].to_drawing())
            for j, n in enumerate(counts)))
        return copies >> Drawing.permutation(perm, [
            self.dom[j:j + 1].to_drawing()
            for j, n in enumerate(counts) for _ in range(n)])

    def to_tree(self) -> dict:
        """
        Serialise a function, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> assert loads(dumps(Function(x @ y, [1, 0, 0])))\\
        ...     == Function(x @ y, [1, 0, 0])
        """
        return dict(factory=factory_name(type(self)),
                    dom=self.dom.to_tree(), func=list(self.func))

    @classmethod
    def from_tree(cls, tree: dict) -> Function:
        return cls(from_tree(tree['dom']), tree['func'])

    def dagger(self) -> Diagram:
        """
        The cofunction of the same list, when the category is also
        comarkov.

        >>> from discopy import closed
        >>> closed.Copy(closed.Ty('x')).dagger()
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: Copy(x) has no dagger: closed.Diagram \
is not a dagger category.
        """
        if not issubclass(self.ar, ComarkovCategory):
            raise AxiomError(messages.NOT_A_DAGGER.format(
                self, factory_name(self.ar)))
        return self.ar.cofunction_factory(self.dom, list(self.func))

    def fuse(self, other: Function) -> Diagram:
        """
        The composite with another function as one function, i.e. the
        identity diagram when it is the identity.

        Parameters:
            other : The function to compose with.

        >>> x = Ty('x')
        >>> assert Copy(x).fuse(Discard(x) @ x) == Id(x)
        """
        assert_iscomposable(self, other)
        return self.ar.from_function(self.func >> other.func, self.dom)

    def tensor(self, *others):
        """
        Tensoring plumbing with plumbing gives one wider function: a run of
        functions and types coalesces into the function on the disjoint
        union of their underlying :class:`finset.Function
        <discopy.python.finset.Function>`, from which point the layers are
        those of an ordinary :meth:`discopy.monoidal.Diagram.tensor`.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Copy(x).tensor(y, Discard(x))\\
        ...     == Function(x @ y @ x, [0, 0, 1])
        """
        funcs, dom, plumbing = [self.func], self.dom, 0
        while plumbing < len(others) and (
                isinstance(others[plumbing], monoidal.Ty)
                or isinstance(others[plumbing], Function)
                and self.ar.layer_factory.is_plumbing_box(others[plumbing])
                or isinstance(others[plumbing], Diagram)
                and not others[plumbing].inside):
            other, plumbing = others[plumbing], plumbing + 1
            typ = other if isinstance(other, monoidal.Ty) else other.dom
            funcs.append(other.func if isinstance(other, Function)
                         else finset.Function.id(len(typ)))
            dom = dom @ typ
        result = self if len(funcs) == 1 else self.ar.from_function(
            funcs[0].tensor(*funcs[1:]), dom)
        rest = others[plumbing:]
        return result if not rest else result.ar.tensor(result, *rest)

    def __rmatmul__(self, other):
        if not isinstance(other, monoidal.Ty):
            return super().__rmatmul__(other)
        func = finset.Function.id(len(other)).tensor(self.func)
        return self.function_factory(other @ self.dom, func)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dom!r}, {list(self.func)})"

    def __str__(self):
        return f"Function({self.dom}, {list(self.func)})"


Layer.plumbing = (monoidal.Ty, Function)


class Permutation(symmetric.Permutation, Box):
    """
    A permutation in a Markov category, i.e. a bijective :class:`Function`,
    registered as a virtual subclass so that it keeps the constructor of
    :class:`discopy.symmetric.Permutation`.

    Parameters:
        dom (monoidal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert isinstance(Swap(x, y), Function)
    >>> assert Swap(x, y) @ Copy(x) == Function(x @ y @ x, [1, 0, 2, 2])
    """
    func = property(lambda self: finset.Function(
        list(self.perm), Nat(len(self.dom)), Nat(len(self.cod))))
    fuse, tensor = Function.fuse, Function.tensor


Function.register(Permutation)


class Swap(Permutation, symmetric.Swap, Box):
    """
    Symmetric swap in a Markov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """


class Copy(Function):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times,
    i.e. the :class:`Function` on one wire that ``n`` outputs read.

    Parameters:
        x : The type to copy.
        n : The number of copies, or the function ``[0] * n`` when a
            :class:`Function` constructs it.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        n = len(n) if isinstance(n, Sized) else index(n)
        if n < 0:
            raise ValueError(f"A copy count is non-negative, got n={n}.")
        name = f"Copy({x}" + ("" if n == 2 else f", {n}") + ")" if n\
            else f"Discard({x})"
        Box.__init__(self, name, dom=x, cod=x ** n,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        n = len(n) if isinstance(n, Sized) else index(n)
        return super().__new__(cls) if n else\
            cls.discard_factory.__new__(cls.discard_factory, x)

    def __getnewargs__(self):
        """
        The arguments of :meth:`__new__`, which a pickle needs.

        >>> from pickle import dumps, loads
        >>> x = Ty('x')
        >>> assert loads(dumps(Copy(x))) == Copy(x)
        >>> assert loads(dumps(Discard(x))) == Discard(x)
        """
        return self.dom, len(self.cod)

    func = property(lambda self: finset.Function.copy(1, len(self.cod)))

    def to_drawing(self):
        return super(Function, self).to_drawing()

    def to_tree(self) -> dict:
        return super(Function, self).to_tree()

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.dom)}, {len(self.cod)})")

    def __str__(self):
        return self.name

    @classmethod
    def from_tree(cls, tree: dict) -> Copy:
        return cls(from_tree(tree['dom']), len(from_tree(tree['cod'])))


class Discard(Copy):
    """
    The discard of an atomic type :code:`x`.

    Parameters:
        x : The type to discard.
    """
    def __init__(self, x: monoidal.Ty, n: int = 0):
        super().__init__(x, n)
        if self.cod:
            raise ValueError(
                f"A discard has no copies, got n={len(self.cod)}.")


class Sum(symmetric.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box.

    Its flags are read off its terms: copying and discarding are linear, so
    that a sum is deterministic when it has at most one term and that term
    is, and affine when it has exactly one and that term is.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.

    Example
    -------
    >>> x = Ty('x')
    >>> f = Box('f', x, x)
    >>> assert Sum((f, )).is_cartesian and not (f + f).is_deterministic
    >>> zero = Sum((), x, x)
    >>> assert zero.is_deterministic and not zero.is_affine
    """
    def __init__(self, terms, dom=None, cod=None):
        super().__init__(terms, dom, cod)
        self.deterministic = len(self.terms) < 2 and all(
            term.is_deterministic for term in self.terms)
        self.affine = len(self.terms) == 1 and self.terms[0].is_affine


class Bubble(symmetric.Bubble, Box):
    """
    A markov bubble is a symmetric bubble and a markov box: a total function
    unless it says otherwise, whatever its args, and it keeps what it says
    through dagger, substitution, rotation and markov functors.

    Parameters:
        args (Diagram) : The diagrams inside the bubble.
        is_deterministic : Whether copying commutes with the bubble.
        is_affine : Whether discarding commutes with the bubble.
        kwargs : Passed to :class:`symmetric.Bubble`.

    Example
    -------
    >>> x = Ty('x')
    >>> sample = Box('sample', x, x, is_deterministic=False)
    >>> noisy = sample.bubble(is_deterministic=False)
    >>> assert sample.bubble().is_deterministic and not noisy.is_deterministic
    >>> assert noisy != sample.bubble() and noisy[::-1][::-1] == noisy
    >>> assert Functor.id(Diagram)(noisy) == noisy
    """
    serialised_attrs = symmetric.Bubble.serialised_attrs + (
        "is_deterministic", "is_affine")

    def __init__(self, *args: Diagram, is_deterministic=True, is_affine=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic, self.affine = is_deterministic, is_affine

    def __repr__(self):
        flags = "".join(f", {name}=False" for name in self.flags())
        return super().__repr__()[:-1] + flags + ")" if flags\
            else super().__repr__()

    def setoid(self):
        return super().setoid() + (self.deterministic, self.affine)

    def to_tree(self) -> dict:
        return {**super().to_tree(), **self.flags()}

    @classmethod
    def from_tree(cls, tree: dict) -> Bubble:
        result = super().from_tree(tree)
        result.deterministic = tree.get("is_deterministic", True)
        result.affine = tree.get("is_affine", True)
        return result


class Functor(symmetric.Functor):
    """
    A Markov functor is a symmetric functor that preserves copies.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------

    We build a functor into python functions.

    >>> x = Ty('x')
    >>> add = Box('add', x @ x, x)
    >>> from discopy import python
    >>> F = Functor({x: int}, {add: lambda a, b: a + b},
    ...             cod=python.Function)
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     doctest="docs/_static/markov/bialgebra.svg")

    .. image:: /_static/markov/bialgebra.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Copy) and hasattr(self.cod, "copy"):
            return self.cod.copy(self(other.dom), len(other.cod))
        if isinstance(other, Function) and hasattr(self.cod, "copy")\
                and not isinstance(other, symmetric.Permutation):
            function = getattr(self.cod, "function", None)\
                or MarkovCategory.function.__func__.__get__(self.cod)
            return function(  # ty: ignore[invalid-argument-type]
                other.func, list(map(self, other.dom)))
        result = super().__call__(other)
        if isinstance(other, Bubble) and isinstance(result, Bubble):
            return other.flagged(result)
        return result


CMap = cmap.CMap[Diagram]

Feedback, Trace, FollowedBy, Head, Tail = feedback.generators(Diagram, Box)
Diagram.box_factory = Box
Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.copy_factory = Copy
Diagram.function_factory = Function
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of Markov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Merge": "discopy.comarkov.Merge"})
