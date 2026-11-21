# -*- coding: utf-8 -*-

"""
The CP* construction of classical-quantum channels over a dagger hypergraph
category, with Selinger's CPM construction as the full subcategory on quantum
wires :cite:p:`CoeckeEtAl16`.

Each wire of the underlying category comes with a choice of algebra: a
:class:`Quantum` wire is the pants algebra on ``x @ x``, so it doubles, and a
:class:`Classical` wire is the copy spider on ``x``, so it stays single. A
:class:`Kraus` box is a channel given by its Kraus map ``k : A -> B @ E`` in
the underlying category, with an environment ``E`` that is discarded, and a
:class:`Doubled` box is one given by its doubled map, e.g. a density matrix.
:meth:`Diagram.double` is the strict monoidal functor into the underlying
category that sends ``k`` to ``k`` beside its conjugate, the environment
joined by a cup, with each wire doubled in place.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Quantum
    Classical
    Ty
    Diagram
    Kraus
    Doubled

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        Q
        C
        nest
        Measure
        Encode
        Discard

Example
-------
>>> from discopy import tensor
>>> x, T = tensor.Dim(2), tensor.Diagram[complex]
>>> plus = Kraus('plus', tensor.Box[complex]('plus', tensor.Dim(1), x,
...                                          [2 ** -.5, 2 ** -.5]))
>>> coin = plus >> Measure(x, T)
>>> coin.double(T).eval().array.real.round(2)
array([0.5, 0.5])
"""

from __future__ import annotations

from collections.abc import Callable

from discopy import cat, frobenius
from discopy.cat import factory, rsubs
from discopy.utils import factory_name, from_tree


class Quantum(frobenius.Wire):
    """
    A quantum wire: a wire ``inside`` of the underlying category with the
    pants algebra on ``inside @ inside``, so that it doubles.

    Parameters:
        inside : An atomic type of the underlying category.
    """
    def __init__(self, inside, *args, **params):
        self.inside = inside
        super().__init__(str(inside), *args, **params)

    def __repr__(self):
        return super().__repr__().replace(
            repr(self.name), repr(self.inside), 1)

    def __str__(self):
        return f"{type(self).__name__[0]}({self.inside})"

    def to_tree(self) -> dict:
        tree = super().to_tree()
        if hasattr(self.inside, "to_tree"):
            tree['inside'] = self.inside.to_tree()
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Quantum:
        wire = super().from_tree(tree)
        if 'inside' in tree:
            wire.inside = from_tree(tree['inside'])
        return wire

    @property
    def double(self):
        """ The image of the wire in the underlying category. """
        return self.inside @ self.inside


class Classical(Quantum):
    """
    A classical wire: a wire ``inside`` of the underlying category with its
    copy spider, so that it stays single.

    Parameters:
        inside : An atomic type of the underlying category.
    """
    @property
    def double(self):
        return self.inside


@factory
class Ty(frobenius.Ty):
    """ A classical-quantum type is a word of quantum and classical wires. """
    generator_factory = Quantum

    @property
    def is_quantum(self) -> bool:
        """ Whether every wire is quantum, i.e. the type is one of CPM. """
        return not any(isinstance(ob, Classical) for ob in self.inside)

    def single(self, unit):
        """
        The underlying type, each wire taken once.

        Parameters:
            unit : The unit of the underlying category.
        """
        return unit.tensor(*(ob.inside for ob in self.inside))


def Q(*xs) -> Ty:  # pylint: disable=invalid-name
    """ The quantum type on some atomic types of the underlying category. """
    return Ty(*map(Quantum, xs))


def C(*xs) -> Ty:  # pylint: disable=invalid-name
    """ The classical type on some atomic types of the underlying category. """
    return Ty(*map(Classical, xs))


def nest(category, typ):
    """
    The permutation from each wire of ``typ`` doubled in place to the nested
    layout ``typ @ typ.r``, the identity on a single wire.

    >>> from discopy.frobenius import Ty, Diagram
    >>> x, y = Ty('x'), Ty('y')
    >>> assert nest(Diagram, x) == Diagram.id(x @ x)
    >>> assert nest(Diagram, x @ y).cod == x @ y @ y @ x
    """
    pairs = typ[:0].tensor(*(x @ x for x in typ))
    return category.permutation(
        [*range(0, len(pairs), 2), *range(len(pairs) - 1, 0, -2)], pairs)


@factory
class Diagram(frobenius.Diagram):
    """
    A classical-quantum diagram: a frobenius diagram whose boxes are channels,
    given by their Kraus map or by their doubled map.
    """
    ob = Ty

    @property
    def is_cpm(self) -> bool:
        """ Whether every wire is quantum, i.e. the diagram is one of CPM. """
        return all(f.dom.is_quantum and f.cod.is_quantum
                   for f in (self, *self.boxes))

    def double(self, category=frobenius.Diagram,
               conjugate: Callable | None = None):
        """
        The image in the underlying category, each quantum wire doubled in
        place and each box sent to its :meth:`Kraus.double`.

        Parameters:
            category : The underlying dagger hypergraph category.
            conjugate : The conjugate of the underlying category, which
                mirrors the wires, ``Diagram.conjugate`` by default.
        """
        return frobenius.Functor(
            ob_map=lambda x: x.inside[0].double,
            ar_map=lambda f: f.double(category, conjugate),
            dom=Diagram, cod=category)(self)


class Box(frobenius.Box, Diagram):
    """
    A classical-quantum box, i.e. a generating channel, equal to another
    when its :attr:`serialised_attrs` are, and serialised by them.
    """
    def setoid(self):
        return super().setoid() + tuple(
            getattr(self, attr) for attr in self.serialised_attrs
            if attr != "*")

    @property
    def maps(self) -> dict:
        """ The arrows of the underlying category the box is given by. """
        return {attr: getattr(self, attr) for attr in self.serialised_attrs
                if isinstance(getattr(self, attr, None), cat.Arrow)}

    @property
    def free_symbols(self):
        return super().free_symbols.union(
            *(f.free_symbols for f in self.maps.values()))

    def subs(self, *args) -> Box:
        if not self.free_symbols:
            return self
        changes = {attr: f.subs(*args) for attr, f in self.maps.items()}
        if "data" in self.serialised_attrs:
            changes["data"] = rsubs(self.data, *args)
        return self.rebuild(**changes)

    def lambdify(self, *symbols, **kwargs):
        if not self.free_symbols & set(symbols):
            return lambda *xs: self
        return lambda *xs: self.subs(list(zip(symbols, xs)))

    def to_tree(self) -> dict:
        return {"factory": factory_name(type(self)), **{
            attr: value.to_tree() if hasattr(value, "to_tree") else value
            for attr in self.serialised_attrs if attr != "*"
            for value in [getattr(self, attr)]}}

    @classmethod
    def from_tree(cls, tree: dict) -> Box:
        return cls(**{
            attr: from_tree(tree[attr]) if isinstance(tree[attr], dict)
            else tree[attr] for attr in cls.serialised_attrs if attr != "*"})


class Kraus(Box):
    """
    A channel given by its Kraus map ``kraus : dom.single() -> cod.single()
    @ env`` in the underlying category, where ``env`` is the environment.

    Parameters:
        name : The name of the channel.
        kraus : The Kraus map, an arrow of the underlying category.
        dom : The domain, quantum on ``kraus.dom`` by default.
        cod : The codomain, quantum on ``kraus.cod`` by default.
        env : The environment, empty by default.
        is_dagger : Whether the channel is the dagger of the one ``kraus``
            gives, from ``cod`` to ``dom``.

    The dagger of a channel keeps its Kraus map, and its double is the
    dagger of the double, as doubling is a dagger functor.
    """
    serialised_attrs = ("name", "kraus", "dom", "cod", "*", "env",
                        "is_dagger")

    def __init__(  # pylint: disable=too-many-arguments,R0917
            self, name, kraus, dom=None, cod=None, env=None,
            is_dagger=False):
        unit = kraus.dom[:0]
        env = unit if env is None else env
        source = Q(*kraus.dom)
        target = Q(*kraus.cod[:len(kraus.cod) - len(env)])
        if is_dagger:
            source, target = target, source
        dom = source if dom is None else dom
        cod = target if cod is None else cod
        source, target = (cod, dom) if is_dagger else (dom, cod)
        assert kraus.dom == source.single(unit)
        assert kraus.cod == target.single(unit) @ env
        self.kraus, self.env = kraus, env
        super().__init__(name, dom, cod, is_dagger=is_dagger)

    def double(self, category=frobenius.Diagram, conjugate=None):
        """
        The CP map of the channel: classical inputs are copied, the Kraus map
        ``k : A -> B @ E`` is put beside its conjugate ``A.r -> E.r @ B.r``,
        which mirrors it, so that the cup on ``E @ E.r`` is in the middle,
        and the two copies of each classical output are merged.

        The mirror is the nested layout ``A @ A.r``, which :func:`nest`
        sends to each wire doubled in place: on a single wire it is the
        identity.
        """
        if self.is_dagger:
            return self.dagger().double(category, conjugate).dagger()
        unit = self.kraus.dom[:0]
        single, env = self.cod.single(unit), self.env
        conjugate = conjugate or (lambda diagram: diagram.conjugate())

        def spiders(typ):
            return category.id(unit).tensor(*(
                category.spiders(1, 2, ob.inside) if isinstance(ob, Classical)
                else category.id(ob.inside @ ob.inside) for ob in typ.inside))

        top = spiders(self.dom) >> nest(category, self.dom.single(unit))
        middle = self.kraus @ conjugate(self.kraus)
        cup = single @ category.cups(env, env.r) @ single.r
        bottom = nest(category, single).dagger() >> spiders(self.cod).dagger()
        return top >> middle >> cup >> bottom


class Doubled(Box):
    """
    A channel given by its doubled map ``doubled : dom.double ->
    cod.double`` in the underlying category, e.g. a density matrix, or by
    the dagger of that map when ``is_dagger``.
    """
    serialised_attrs = ("name", "doubled", "dom", "cod", "*", "is_dagger")

    def __init__(  # pylint: disable=too-many-arguments,R0917
            self, name, doubled, dom, cod, is_dagger=False):
        self.doubled = doubled
        super().__init__(name, dom, cod, is_dagger=is_dagger)

    def double(self, category=frobenius.Diagram, conjugate=None):
        return self.doubled.dagger() if self.is_dagger else self.doubled


def Measure(x, category=frobenius.Diagram):  # pylint: disable=invalid-name
    """ Measure a quantum wire into a classical one. """
    return Kraus('Measure', category.id(x), Q(*x), C(*x))


def Encode(x, category=frobenius.Diagram):  # pylint: disable=invalid-name
    """ Encode a classical wire into a quantum one. """
    return Kraus('Encode', category.id(x), C(*x), Q(*x))


def Discard(typ: Ty, category=frobenius.Diagram):  # pylint: disable=C0103
    """ Discard a classical-quantum type, i.e. take it as the environment. """
    single = typ.single(category.id().dom)
    return Kraus('Discard', category.id(single), typ, Ty(), env=single)
