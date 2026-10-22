# -*- coding: utf-8 -*-

"""
The free compact category on a symmetric traced category, or more generally the
free ribbon category on a balanced traced category.

Concretely, this is a "glorification of the construction of the integers from
the natural numbers". This so-called Int-construction first appeared in Joyal,
Street & Verity :cite:p:`JoyalEtAl96`. It is sometimes called the "geometry of
interaction" construction, see :cite:t:`Abramsky96`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Diagram
    NamedGeneric

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        Int

Example
-------

>>> from discopy.grammar import pregroup
>>> from discopy.grammar.pregroup import Word, Cup, Diagram, Functor
>>> s, n = map(pregroup.Ty, "sn")
>>> Alice, loves, Bob\\
...     = Word('Alice', n), Word('loves', n.r @ s @ n.l), Word('Bob', n)
>>> who = Word('who', n.r @ n @ (n.r @ s).l)
>>> noun_phrase = Alice @ who @ loves @ Bob\\
...     >> Cup(n, n.r) @ n @ Diagram.cups((n.r @ s).l, n.r @ s) @ Cup(n.l, n)

>>> from discopy.frobenius import Ty as T, Diagram as D, Box, Swap
>>> S, N = map(T, "SN")
>>> F = Functor(
...     ob_map={s: Ty[T](S), n: Ty[T](N)},
...     ar_map={Alice: Box('A', T(), N),
...         who: Box('W', S @ N, N @ N),
...         loves: Box('L', N @ N, S),
...         Bob: Box('B', T(), N)},
...     cod=Int(D))
>>> image = F(noun_phrase).inside.to_hypergraph().interchange(1, 3)\\
...     .to_diagram().interchange(1, 2).naturality(2, left=False)

>>> from discopy.monoidal import Equation
>>> Equation(noun_phrase, image, symbol="$\\\\mapsto$").draw(
...     doctest="docs/_static/int/alice-loves-interaction.svg")

.. image:: /_static/int/alice-loves-interaction.svg
    :align: center
"""

from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass
from functools import wraps

from discopy import (
    balanced,
    pivotal,
    ribbon,
    messages
)
from discopy.abc import (
    NamedGeneric, Pregroup, RibbonCategory, SymmetricCategory,
    TracedCategory)
from discopy.cat import assert_iscomposable
from discopy.python import finset
from discopy.utils import (
    AxiomError, factory, classproperty, unbiased, assert_isinstance,
    factory_name, get_origin)


@dataclass
class Ty(Pregroup, NamedGeneric['natural']):
    """
    An integer type is a pair of :attr:`natural` types.

    Parameters:
        positive : The positive half of the type.
        negative : The negative half of the type.

    Note
    ----
    Integer types are parameterised by natural types, e.g.

    >>> from discopy.abc import Nat
    >>> assert Ty == Ty[pivotal.Ty] and Ty[Nat].natural == Nat

    The prefix operator ``-`` reverses positive and negative, e.g.

    >>> x, y, z = (Ty[Nat](Nat(n)) for n in (1, 2, 3))
    >>> assert x @ -y @ z == Ty[Nat](Nat(1) @ Nat(3), Nat(2))

    The negative halves of a tensor come in reverse order, as the duals of
    a rigid category do; :attr:`negatives` is the hook that orders them,
    which :class:`lens.Ty <discopy.lens.Ty>` overrides to keep them
    side by side.

    >>> from discopy.pivotal import Ty as T
    >>> x, y = Ty(T("x"), T("x'")), Ty(T("y"), T("y'"))
    >>> assert x @ y == Ty(T("x") @ T("y"), T("y'") @ T("x'"))
    """
    natural = pivotal.Ty
    negatives = staticmethod(reversed)

    positive: natural
    negative: natural

    def __init__(self, positive: natural = None, negative: natural = None):
        positive, negative = (
            self.natural() if x is None else x for x in (positive, negative))
        positive, negative = (
            x if isinstance(x, type(self).natural)
            else type(self).natural.cast(x)
            for x in (positive, negative))
        self.positive, self.negative = positive, negative

    def __iter__(self):
        yield self.positive
        yield self.negative

    def __bool__(self):
        return bool(self.positive) or bool(self.negative)

    def __repr__(self):
        factory, natural = map(
            factory_name, (get_origin(type(self)), self.natural))
        return f"{factory}[{natural}]"\
            f"(positive={self.positive!r}, negative={self.negative!r})"

    def __str__(self):
        try:
            return " @ ".join(list(map(str, self.positive)) + [
                f"-{x}" for x in self.negatives(self.negative)])
        except TypeError:  # e.g. when Ty.natural == int
            return repr(self)

    @classmethod
    def unit(cls) -> Ty:
        return cls()

    def tensor(self, *others: Ty):
        if any(not isinstance(other, type(self)) for other in others):
            return NotImplemented
        unit = type(self).natural()
        positive = unit.tensor(*(x.positive for x in (self, ) + others))
        negative = unit.tensor(
            *(x.negative for x in self.negatives((self, ) + others)))
        return type(self)(positive, negative)

    __matmul__ = tensor

    def __rmatmul__(self, other):
        """ A type does not whisker: the tensor with anything but a type of
        the same natural numbers is undefined. """
        return NotImplemented

    def __neg__(self):
        positive, negative = self
        return type(self)(negative, positive)

    l = r = property(__neg__)

    def split(self, n: int | tuple[int, int] = 1, left: bool = False
              ) -> tuple[Ty, Ty]:
        """
        The pair of types that tensor to ``self``, the second with ``n`` atoms
        on each half, or the first when ``left``.

        Parameters:
            n : The number of atoms of each half to split off, or one number
                per half.
            left : Whether to split them off the left rather than the right.

        Note
        ----
        An integer type is not a sequence of atoms: positive and negative
        atoms commute, i.e. ``x @ -y == -y @ x``, so there is no slicing.
        The negative halves of the two factors are tensored in the order of
        :attr:`negatives`, which is where the negative half is cut.

        >>> from discopy.pivotal import Ty as T
        >>> x, y = Ty(T("x"), T("x'")), Ty(T("y"), T("y'"))
        >>> assert (x @ y).split() == (x, y) == (x @ y).split(left=True)
        >>> assert (x @ y).split((1, 0))\\
        ...     == (Ty(T("x"), T("y'") @ T("x'")), Ty(T("y")))
        """
        positive, negative = self
        n_pos, n_neg = (n, n) if isinstance(n, int) else n
        if not 0 <= n_pos <= len(positive) or not 0 <= n_neg <= len(negative):
            raise ValueError(messages.NOT_SPLITTABLE.format(n, self))
        i, j = (n_pos, n_neg) if left else (
            len(positive) - n_pos, len(negative) - n_neg)
        cut, _ = self.negatives((j, len(negative) - j))
        first_negative, second_negative = self.negatives(
            (negative[:cut], negative[cut:]))
        return (type(self)(positive[:i], first_negative),
                type(self)(positive[i:], second_negative))


@factory
@dataclass
class Diagram(RibbonCategory, NamedGeneric['natural']):
    """
    An integer diagram from ``x`` to ``y`` is a :attr:`natural` diagram
    from ``x.positive @ y.negative`` to ``x.negative @ y.positive``.

    Parameters:
        inside : The natural diagram inside.
        dom : The domain of the diagram, i.e. its input.
        cod : The codomain of the diagram, i.e. its output.

    Note
    ----
    By default we take ``natural = ribbon.Diagram`` but this can be any class
    with the methods for a balanced traced category, or for a symmetric traced
    one, see :meth:`is_symmetric`. For example, the category of boolean
    matrices with the direct sum has a trace given by reflexive transitive
    closure. We can use it to check the snake equations:

    >>> from discopy.abc import Nat
    >>> from discopy.matrix import Matrix
    >>> T, D = Ty[Nat], Diagram[Matrix[bool]]
    >>> two = T(Nat(2), Nat(2))
    >>> assert D.id(two).transpose()\\
    ...     == D.id(two)\\
    ...     == D.id(two).transpose(left=True)

    Python functions with tuples as tensor have neither dagger nor twist,
    their integer diagrams are the lenses of :mod:`discopy.lens`:

    >>> from discopy.python import Function
    >>> R = Ty[Function.ob]((float, ), (float, ))
    >>> inside = Function(
    ...     lambda a, da: (a * a, 2 * a * da), (float, float), (float, float))
    >>> square = Diagram[Function](inside, R, R)
    >>> assert (square @ square).inside(2., 3., 1., 1.) == (4., 9., 6., 4.)
    >>> assert (square >> square).dom == R == (square >> square).cod
    """
    ar = classproperty(lambda cls: cls)
    natural = ribbon.Diagram

    ob = classproperty(lambda cls: Ty[cls.natural.ob])

    inside: natural
    dom: Ty
    cod: Ty

    def __init__(self, inside: natural, dom: Ty, cod: Ty):
        assert_isinstance(inside, self.natural)
        if inside.dom != dom.positive @ cod.negative:
            raise ValueError(messages.WRONG_DOM.format(
                dom.positive @ cod.negative, inside.dom))
        if inside.cod != cod.positive @ dom.negative:
            raise ValueError(messages.WRONG_COD.format(
                cod.positive @ dom.negative, inside.cod))
        self.inside, self.dom, self.cod = inside, dom, cod

    @classmethod
    def is_symmetric(cls) -> bool:
        """
        Whether the integer diagrams are symmetric, i.e. compact rather than
        ribbon: when the :attr:`natural` category is, in which case its braid
        is its own inverse and its twist is the identity, so that it need not
        have a dagger nor a twist.
        """
        return issubclass(cls.natural, SymmetricCategory)

    @classmethod
    def natural_braid(
            cls, left: natural.ob, right: natural.ob, inverse: bool = False
            ) -> natural:
        """
        The braid of the :attr:`natural` category or its inverse, i.e. the
        dagger of the braid, or the braid the other way around when
        :meth:`is_symmetric`.

        Parameters:
            left : The natural type on the left of the braid.
            right : The natural type on the right of the braid.
            inverse : Whether to take the inverse, from ``right @ left``.
        """
        if not inverse:
            return cls.natural.braid(left, right)
        if cls.is_symmetric():
            return cls.natural.braid(right, left)
        return cls.natural.braid(left, right).dagger()

    @classmethod
    def natural_twist(cls, dom: natural.ob, inverse: bool = False) -> natural:
        """
        The twist of the :attr:`natural` category or its inverse, i.e. the
        dagger of the twist, or the identity when :meth:`is_symmetric`.

        Parameters:
            dom : The natural type to twist.
            inverse : Whether to take the inverse twist.
        """
        if cls.is_symmetric():
            return cls.natural.id(dom)
        twist = cls.natural.twist(dom)
        return twist.dagger() if inverse else twist

    @unbiased
    def then(self, other: Diagram):
        """
        The composition of two integer diagrams.

        Parameters:
            other : The other diagram with which to compose.

        Example
        -------

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x0, y0, z0 = map(T, [obj + "0" for obj in "xyz"])
        >>> x1, y1, z1 = map(T, [obj + "1" for obj in "xyz"])
        >>> X, Y, Z = Ty[T](x0, x1), Ty[T](y0, y1), Ty[T](z0, z1)
        >>> f = Diagram[D](B('f', x0 @ y1, y0 @ x1), X, Y)
        >>> g = Diagram[D](B('g', y0 @ z1, z0 @ y1), Y, Z)
        >>> (f >> g).draw(doctest='docs/_static/int/composition.svg')

        .. image:: /_static/int/composition.svg
            :align: center

        Note
        ----

        In the compact case, composition is equivalent to symmetric feedback:

        >>> from discopy.interaction import *
        >>> from discopy.compact import Ty, Box, Swap, Cup, Cap
        >>> f_, g_ = Box('f', x0 @ y1, x1 @ y0), Box('g', y0 @ z1, y1 @ z0)
        >>> caps = (x0 @ Cap(y1, y1.r) @ Cap(y0.r, y0) @ z1).foliation()
        >>> cups = (x1 @ Cup(y0, y0.r) @ Cup(y1.r, y1) @ z0).foliation()
        >>> symmetric_feedback =\\
        ...     caps >> (f_ @ Swap(y1.r, y0.r) @ g_).foliation() >> cups
        >>> symmetric_feedback.draw(
        ...     doctest='docs/_static/int/symmetric-feedback.svg')

        .. image:: /_static/int/symmetric-feedback.svg
            :align: center
        """
        assert_iscomposable(self, other)
        x, u = self.dom
        y, v = self.cod
        z, w = other.cod
        braid = self.natural_braid
        dom, cod = type(self.dom)(x, u), type(self.cod)(z, w)
        inside = (
            x @ braid(w, v)
            >> self.inside @ w
            >> y @ braid(w, u, inverse=True)
            >> other.inside @ u
            >> z @ braid(v, u)).trace(n=v if isinstance(v, int) else len(v))
        return type(self)(inside, dom, cod)

    @classmethod
    def id(cls, dom: Ty = None) -> Diagram:
        """
        The identity on an integer type.

        Parameters:
            dom : The integer type on which to take the identity.

        Example
        -------

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> (Diagram[D].id(x @ -u) >> f).draw(
        ...     doctest='docs/_static/int/idl.svg')

        .. image:: /_static/int/idl.svg
            :align: center

        >>> (f >> Diagram[D].id(y @ -v)).draw(
        ...     doctest='docs/_static/int/idr.svg')

        .. image:: /_static/int/idr.svg
            :align: center
        """
        dom = Ty[cls.natural.ob]() if dom is None else dom
        positive, negative = dom
        inside = cls.natural.id(positive) @ cls.natural_twist(negative)
        return cls(inside, dom, dom)

    @unbiased
    def tensor(self, other):
        """
        The tensor of two integer diagrams.

        Parameters:
            other : The other diagram to tensor.

        Example
        -------

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> x_, y_, u_, v_ = map(lambda x: Ty[T](x + '_'), "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> f_ = Diagram[D](
        ...     B('f_', T('x_', 'v_'), T('y_', 'u_')), x_ @ -u_, y_ @ -v_)
        >>> (f @ f_).draw(doctest='docs/_static/int/tensor.svg')

        .. image:: /_static/int/tensor.svg
            :align: center
        """
        x, u, y, v = tuple(self.dom) + tuple(self.cod)
        x_, _, _, v_ = tuple(other.dom) + tuple(other.cod)
        braid = self.natural_braid
        inside = braid(x, x_) @ braid(v, v_, inverse=True)\
            >> x_ @ self.inside @ v_\
            >> braid(y, x_, inverse=True) @ braid(v_, u, inverse=True)\
            >> y @ other.inside @ u
        return type(self)(inside, self.dom @ other.dom, self.cod @ other.cod)

    @classmethod
    def braid(cls, left: Ty, right: Ty) -> Diagram:
        """
        The braid of integer diagrams is given by the following diagram:

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, u, y, v = map(Ty[T], "xuyv")
        >>> Diagram.braid(x @ -u, y @ -v).draw(
        ...     doctest="docs/_static/int/braid.svg")

        .. image:: /_static/int/braid.svg
            :align: center

        Parameters:
            left : The left input of the braid.
            right : The right input of the braid.
        """
        braid, twist = cls.natural_braid, cls.natural_twist
        x, u = left
        y, v = right
        braids = braid(x, y) @ braid(v, u, inverse=True) >> y @ (
            braid(v, x, inverse=True) >> braid(x, v, inverse=True)) @ u
        twists = y @ x @ twist(v, inverse=True) @ twist(u, inverse=True)
        return cls(braids >> twists, left @ right, right @ left)

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[Ty]) -> Diagram:
        xs = finset.Permutation(xs, len(doms))
        dom = cls.ob.tensor(*doms) if doms else cls.ob.unit()
        if xs.is_identity:
            return cls.id(dom)

        base = cls.natural
        positive = [ob.positive for ob in doms]
        negative = [doms[i].negative for i in reversed(xs)]
        inside = (
            base.permutation(xs, positive) @
            base.permutation(xs.rotate(), negative)
        )
        cod = cls.ob.tensor(*(doms[i] for i in xs)) if doms else cls.ob.unit()
        return cls(inside, dom, cod)

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer cups are given by natural identities.

        Parameters:
            left : The left-hand side of the cups.
            right : The right-hand side of the cups.

        Example
        -------

        This is what the snake equations look like:

        >>> from discopy.monoidal import Equation
        >>> x = Ty('x')
        >>> Equation(
        ...     Diagram.caps(x, -x) @ x >> x @ Diagram.cups(-x, x),
        ...     Diagram.id(x),
        ...     x @ Diagram.caps(-x, x) >> Diagram.cups(x, -x) @ x).draw(
        ...         doctest="docs/_static/int/int-snake-equations.svg")

        .. image:: /_static/int/int-snake-equations.svg
            :align: center
        """
        left.assert_isadjoint(right)
        inside = cls.natural.id(left.positive @ left.negative)
        return cls(inside, left @ right, type(left)())

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer caps are given by natural identities.

        Parameters:
            left : The left-hand side of the caps.
            right : The right-hand side of the caps.
        """
        right.assert_isadjoint(left)
        inside = cls.natural.id(left.positive @ left.negative)
        return cls(inside, type(left)(), left @ right)

    def dagger(self):
        """
        The dagger of an integer diagram is given by the dagger of its inside.

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> from discopy.monoidal import Equation
        >>> Equation(f, f[::-1], symbol="$\\\\mapsto$").draw(
        ...     doctest="docs/_static/int/dagger.svg")

        .. image:: /_static/int/dagger.svg
            :align: center
        """
        return type(self)(self.inside.dagger(), self.cod, self.dom)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise IndexError

    def draw(self, **params):
        """ The drawing of an integer diagram is the drawing of its inside. """
        return self.inside.draw(**params)

    def simplify(self):
        """
        Simplify by going back and forth to :class:`Hypergraph`.

        Example
        -------
        >>> from discopy import frobenius
        >>> x = Ty[frobenius.Ty]('x')
        >>> D = Diagram[frobenius.Diagram]
        >>> left_snake = D.id(-x).transpose(left=True)
        >>> right_snake = D.id(-x).transpose(left=False)
        >>> assert left_snake.simplify() == D.id(x) == right_snake.simplify()

        >>> from discopy.monoidal import Equation
        >>> Equation(left_snake, Equation(
        ...     D.id(x), right_snake, symbol="$\\\\leftarrow$"),
        ...         symbol="$\\\\rightarrow$").draw(
        ...             doctest="docs/_static/int/simplify.svg")

        .. image:: /_static/int/simplify.svg
            :align: center
        """
        return type(self)(self.inside.simplify(), self.dom, self.cod)

    @wraps(balanced.Diagram.naturality)
    def naturality(self, i: int, left=True, down=True, braid=None) -> Diagram:
        return type(self)(
            self.inside.naturality(i, left, down, braid), self.dom, self.cod)

    def trace(self, n: int | tuple[int, int] = 1, left: bool = False
              ) -> Diagram:
        """
        The trace of an integer diagram over the last ``n`` atoms of each half
        of its domain and codomain, or the first when ``left``, is the natural
        trace of its inside over the two halves at once, after the braids that
        bring them side by side on the right.

        Parameters:
            n : The number of atoms of each half to trace over, or one number
                per half.
            left : Whether to trace the atoms on the left or right.

        Example
        -------
        >>> from discopy.compact import Ty as T, Box, Diagram as D
        >>> a, a_, u, u_, b, b_ = map(T, ["a", "a'", "u", "u'", "b", "b'"])
        >>> A, U, B = Ty[T](a, a_), Ty[T](u, u_), Ty[T](b, b_)
        >>> f = Diagram[D](
        ...     Box('f', a @ u @ u_ @ b_, b @ u @ u_ @ a_), A @ U, B @ U)
        >>> assert f.trace().dom == A and f.trace().cod == B
        >>> f.trace().draw(doctest="docs/_static/int/trace.svg")

        .. image:: /_static/int/trace.svg
            :align: center

        Note
        ----
        Integer diagrams are compact, so the trace is also the feedback loop
        made of cups and caps, up to the axioms of traced categories, which
        the hypergraph decides when the natural category is symmetric:

        >>> feedback = A @ Diagram[D].caps(U, -U) >> f @ -U\\
        ...     >> B @ Diagram[D].cups(U, -U)
        >>> assert f.trace().inside.to_hypergraph()\\
        ...     == feedback.inside.to_hypergraph()
        """
        (dom, traced), (cod, traced_cod) = (
            x.split(n, left)[::-1] if left else x.split(n, left)
            for x in (self.dom, self.cod))
        if traced != traced_cod:
            raise AxiomError(messages.NOT_TRACEABLE.format(traced, traced_cod))
        if not traced:
            return self
        (a, a_), (b, b_), (u, u_) = dom, cod, traced
        braid = self.natural_braid
        inside = braid(a @ b_, u) @ u_ >> self.inside >> braid(u, b @ a_) @ u_\
            if left else a @ braid(b_, u @ u_) >> self.inside >> b @ braid(
                u @ u_, a_)
        return type(self)(inside.trace(len(u @ u_)), dom, cod)

    def split_memory(
            self, n: int | tuple[int, int] = None, left: bool = False, *,
            dom: Ty = None, cod: Ty = None, mem: Ty = None
            ) -> tuple[Ty, Ty, Ty]:
        """
        The domain, codomain and memory of a trace, read off either form of
        :meth:`discopy.abc.FeedbackCategory.feedback` with :meth:`Ty.split`,
        since an integer type is not a sequence: ``n`` counts the atoms of
        each half, or one number per half, one by default, and the memory is
        ``mem``, or what ``cod`` leaves of the codomain.

        Example
        -------
        >>> from discopy.compact import Ty as T, Box, Diagram as D
        >>> a, a_, u, u_ = map(T, ["a", "a'", "u", "u'"])
        >>> A, U = Ty[T](a, a_), Ty[T](u, u_)
        >>> box = Box('f', a @ u @ u_ @ a_, a @ u @ u_ @ a_)
        >>> f = Diagram[D](box, A @ U, A @ U)
        >>> assert f.split_memory() == f.split_memory(mem=U) == (A, A, U)
        >>> assert f.trace(mem=U) == f.trace() == f.feedback(cod=A)
        """
        if n is not None and (dom, cod, mem) != (None, None, None):
            raise TypeError(messages.FEEDBACK_FORMS)
        if mem is not None:
            n = self.memory_size(mem)
        elif cod is not None:
            n = tuple(len(x) - len(y) for x, y in zip(self.cod, cod))
        (rest_dom, _), (rest_cod, traced) = (
            x.split(1 if n is None else n, left)[::-1] if left
            else x.split(1 if n is None else n, left)
            for x in (self.dom, self.cod))
        return (rest_dom if dom is None else dom,
                rest_cod if cod is None else cod,
                traced if mem is None else mem)

    def memory_size(self, mem: Ty) -> tuple[int, int]:
        """
        The number of atoms of each half of a memory, as :meth:`trace`
        counts them.

        Parameters:
            mem : The memory.
        """
        return tuple(map(len, mem))

    @classmethod
    def twist(cls, dom: Ty) -> Diagram:
        """
        The twist of an integer type is the trace of its braid, as in
        :mod:`discopy.ribbon`, i.e. the identity when :meth:`is_symmetric`.

        Parameters:
            dom : The integer type to twist.

        Example
        -------
        >>> from discopy import compact
        >>> x = Ty('x')
        >>> assert Diagram.twist(x).dom == x == Diagram.twist(x).cod
        >>> X = Ty[compact.Ty]('x')
        >>> assert Diagram[compact.Diagram].twist(X)\\
        ...     == Diagram[compact.Diagram].id(X)
        """
        if cls.is_symmetric():
            return cls.id(dom)
        return cls.braid(dom, dom).trace(tuple(map(len, dom)))

    def curry(self, n: int | tuple[int, int] = 1, left: bool = True
              ) -> Diagram:
        """
        The currying of an integer diagram is given by caps, as in any rigid
        category, over the last ``n`` atoms of each half of its domain, or the
        first when not ``left``.

        Parameters:
            n : The number of atoms of each half to curry, or one number per
                half.
            left : Whether to curry on the left or right.

        Example
        -------
        >>> from discopy.compact import Ty as T, Box, Diagram as D
        >>> x, x_, y, y_ = map(T, ["x", "x'", "y", "y'"])
        >>> X, Y = Ty[T](x, x_), Ty[T](y, y_)
        >>> f = Diagram[D](Box('f', x @ y_, y @ x_), X, Y)
        >>> assert f.curry().dom == Ty[T]() and f.curry().cod == Y @ -X
        >>> assert f.curry().uncurry().inside.to_hypergraph()\\
        ...     == f.inside.to_hypergraph()
        """
        base, exponent = self.dom.split(n) if left\
            else self.dom.split(n, left=True)[::-1]
        if not exponent:
            return self
        if left:
            return base @ self.caps(exponent, -exponent) >> self @ -exponent
        return self.caps(-exponent, exponent) @ base >> -exponent @ self

    def base_and_exponent(self, n: int | tuple[int, int], left: bool
                          ) -> tuple[Ty, Ty]:
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        codomain where the exponent comes dualised, i.e. with its halves
        exchanged.

        Parameters:
            n : The number of atoms of each half of the exponent, or one
                number per half.
            left : Whether to uncurry on the left or right.
        """
        n = n if isinstance(n, int) else n[::-1]
        base, exponent = self.cod.split(n) if left\
            else self.cod.split(n, left=True)[::-1]
        return base, -exponent

    def uncurry(self, n: int | tuple[int, int] = 1, left: bool = True
                ) -> Diagram:
        """
        The uncurrying of an integer diagram is its evaluation, as in any
        rigid category, over the exponent at the end of its codomain, or at
        the start when not ``left``.

        Parameters:
            n : The number of atoms of each half of the exponent, or one
                number per half.
            left : Whether to uncurry on the left or right.
        """
        base, exponent = self.base_and_exponent(n, left)
        if not exponent:
            return self
        return self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)

    def rotate(self, left: bool = False) -> Diagram:
        """
        The rotation of an integer diagram is its :meth:`transpose`, since it
        has no box to rotate.

        Parameters:
            left : Whether to rotate left or right.
        """
        return self.transpose(left)

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))
    conjugate = pivotal.Diagram.conjugate
    boxes = property(lambda self: self.inside.boxes)
    to_drawing = lambda self: self.inside.to_drawing()


def Int(category: TracedCategory) -> RibbonCategory:
    """
    The Int construction, i.e. the free ribbon category on a given balanced
    traced `category`, or the free compact category on a symmetric traced
    one, with :class:`interaction.Ty` as objects and
    :class:`interaction.Diagram` as arrows.

    Parameters:
        category : A balanced or symmetric traced category, i.e. an arrow
            type.

    Example
    -------
    >>> from discopy.ribbon import Diagram as D
    >>> assert Int(D) == Diagram[D]
    """
    return Diagram[category]


Id = Diagram.id
