# -*- coding: utf-8 -*-

"""
The free delay category, i.e. a category with an endofunctor ``delay``, see
:class:`~discopy.abc.DelayCategory`: an object and a box carry a time step,
the number of times they have been delayed, and the delay of an arrow is
that of its boxes. It needs no tensor: the feedback is that of
:mod:`~discopy.feedback`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Arrow
    Box
    Wait
    Functor

Example
-------
>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g = Box('f', x, y), Box('g', y, z)
>>> assert (f >> g).delay() == f.delay() >> g.delay()
>>> assert Arrow.id(x).delay() == Arrow.id(x.delay())
>>> assert f.delay(2) == f.delay().delay() != f
>>> print((f >> g).delay())
f.d >> g.d

The wait is a natural transformation from the identity to the delay, a
primitive :class:`Wait` at each object, see
:meth:`~discopy.abc.DelayCategory.wait`.

>>> assert Arrow.wait(x) == Wait(x) and Wait(x).cod == x.d
>>> print(f >> Arrow.wait(y))
f >> Wait(y)
>>> print(Arrow.wait(x) >> f.d)
Wait(x) >> f.d
"""

from __future__ import annotations

from typing import ClassVar

from discopy import cat, messages
from discopy.abc import DelayCategory
from discopy.cat import factory
from discopy.utils import assert_isinstance, factory_name, from_tree


def str_delayed(time_step: int) -> str:
    """ The suffix that prints a time step, e.g. ``.d.d`` for two. """
    return time_step * ".d" if time_step <= 3 else f".delay({time_step})"


class Ob(cat.Ob):
    """
    An object with a time step, the number of times it has been delayed.

    Parameters:
        name : The name of the object.
        time_step : The number of times the object has been delayed.

    Example
    -------
    >>> from discopy import feedback
    >>> x = Ob('x')
    >>> assert x.delay(2) == Ob('x', time_step=2) != x
    >>> assert eval(repr(x.delay())) == x.delay() and str(x.delay()) == 'x.d'
    >>> assert x.delay(2).reset() == x
    """
    def __init__(self, name: str = "", time_step: int = 0):
        assert_isinstance(time_step, int)
        if time_step < 0:
            raise NotImplementedError
        super().__init__(name)
        self.time_step = time_step

    def delay(self, n_steps: int = 1) -> Ob:
        """ The delay of an object, ``n_steps`` time steps on. """
        return type(self)(self.name, self.time_step + n_steps)

    d = property(lambda self: self.delay())

    def reset(self) -> Ob:
        """ The object at time step zero, used in :class:`Functor`. """
        return type(self)(self.name)

    def atoms(self) -> list[Ob]:
        """ The atom of an object, at time step zero. """
        return [self.reset()]

    def __eq__(self, other):
        return super().__eq__(other) and self.time_step == other.time_step

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    def to_tree(self) -> dict:
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Ob:
        return cls(tree['name'], tree.get('time_step', 0))


@factory
class Arrow(cat.Arrow, DelayCategory):
    """
    An arrow in the free delay category, a :class:`~discopy.cat.Arrow` whose
    delay is that of its boxes.

    Parameters:
        inside (tuple[Box, ...]) : The boxes inside the arrow.
        dom (Ob) : The domain of the arrow, i.e. its input.
        cod (Ob) : The codomain of the arrow, i.e. its output.
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    wait_factory: ClassVar[type["Wait"]]
    ob = Ob
    wait_naturality = DelayCategory.wait_naturality.failing(
        messages.FREE_WAIT)

    def delay(self, n_steps: int = 1) -> Arrow:
        """ The delay of an arrow, that of each of its boxes. """
        return type(self)(
            tuple(box.delay(n_steps) for box in self.inside),
            self.dom.delay(n_steps), self.cod.delay(n_steps))

    d = property(lambda self: self.delay())

    @classmethod
    def wait(cls, dom: Ob) -> Arrow:
        """ The primitive :class:`Wait` on an object. """
        return cls.wait_factory(dom)


class Box(cat.Box, Arrow):
    """
    A box with a time step, the number of times it has been delayed.

    Parameters:
        name (str) : The name of the box.
        dom (Ob) : The domain of the box, i.e. its input.
        cod (Ob) : The codomain of the box, i.e. its output.
        time_step (int) : The number of times the box has been delayed.

    Example
    -------
    >>> from discopy import feedback
    >>> f = Box('f', Ob('x'), Ob('y')).delay()
    >>> assert f.time_step == 1 and f.dom == Ob('x').delay()
    >>> assert eval(repr(f)) == f and f.reset() == Box('f', Ob('x'), Ob('y'))
    >>> assert f.dagger().dagger() == f and f.dagger().time_step == 1
    """
    serialised_attrs = cat.Box.serialised_attrs + ("time_step", )

    def __init__(self, name: str, dom: Ob, cod: Ob, *args,
                 time_step: int = 0, **params):
        assert_isinstance(time_step, int)
        if time_step < 0:
            raise NotImplementedError
        self.time_step = time_step
        super().__init__(name, dom, cod, *args, **params)

    def delay(self, n_steps: int = 1) -> Box:
        return self.rebuild(
            dom=self.dom.delay(n_steps), cod=self.cod.delay(n_steps),
            time_step=self.time_step + n_steps)

    def reset(self) -> Box:
        """ The box at time step zero, used in :class:`Functor`. """
        return self.delay(-self.time_step)

    def atoms(self) -> list:
        """ The atoms of a box, each at time step zero. """
        return [atom.reset() for atom in super().atoms()]

    def setoid(self):
        return (super().setoid(), self.time_step)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def __str__(self):
        if self.args and self.time_step:
            return str(self.reset()) + str_delayed(self.time_step)
        return super().__str__() + str_delayed(self.time_step)

    def to_tree(self) -> dict:
        tree = super().to_tree()
        if self.time_step:
            tree['time_step'] = self.time_step
        return tree

    @classmethod
    def from_tree(cls, tree: dict):
        return super().from_tree(tree).rebuild(
            time_step=tree.get('time_step', 0))


class Wait(Box):
    """
    The wait on an object ``x``, from ``x`` to ``x.d``: the component at
    ``x`` of the natural transformation from the identity to the delay.

    Parameters:
        x : The object to wait on.

    Example
    -------
    >>> from discopy import feedback
    >>> x = Ob('x')
    >>> assert Wait(x).delay() == Wait(x.d) and eval(repr(Wait(x))) == Wait(x)
    """
    serialised_attrs = ("arg", "is_dagger")
    arg: Ob = None  # ty: ignore[invalid-assignment]  # shadows Box.arg

    def __init__(self, arg: Ob, is_dagger: bool = False):
        self.arg = arg
        dom, cod = (arg.delay(), arg) if is_dagger else (arg, arg.delay())
        Box.__init__(self, f"Wait({arg})", dom, cod, is_dagger=is_dagger)

    def delay(self, n_steps: int = 1) -> Wait:
        return type(self)(self.arg.delay(n_steps), self.is_dagger)

    def reset(self) -> Wait:
        return self

    def dagger(self) -> Wait:
        return type(self)(self.arg, not self.is_dagger)

    def __repr__(self):
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self)) + f"({self.arg!r}{is_dagger})"

    def __str__(self):
        return self.name + (".dagger()" if self.is_dagger else "")

    def to_tree(self) -> dict:
        tree = {'factory': factory_name(type(self)), 'arg': self.arg.to_tree()}
        if self.is_dagger:
            tree['is_dagger'] = True
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Wait:
        return cls(from_tree(tree['arg']), tree.get('is_dagger', False))


class Functor(cat.Functor):
    """
    A delay functor is a functor that preserves the delay: an object or a box
    with a time step goes to the delay of the image of its reset, through the
    delay of the codomain, the identity when it has none.

    Parameters:
        ob_map (Mapping[Ob, Ob]) : Map from :class:`Ob` to :code:`cod.ob`.
        ar_map (Mapping[Box, Arrow]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Arrow` by default.

    Example
    -------
    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y)
    >>> F = Functor({x: y, y: x}, {f: f[::-1]})
    >>> assert F(f.delay(2)) == F(f).delay(2) == f[::-1].delay(2)

    Into a category whose delay is the identity, a time step goes away.

    >>> from discopy.python import Function
    >>> G = Functor({x: int, y: int}, {f: lambda n: n + 1}, cod=Function)
    >>> assert G(f.delay())(41) == 42
    """
    dom = cod = Arrow

    def __call__(self, other):
        if isinstance(other, Wait) and hasattr(self.cod, "wait") and (
                self(other.dom) == self(other.cod) or getattr(
                    self.cod.wait, "__func__", None)
                is not DelayCategory.wait.__func__):
            result = self.cod.wait(self(other.arg))
            return result.dagger() if other.is_dagger else result
        if isinstance(other, (Ob, Box)) and other.time_step:
            result = self(other.reset())
            delay = getattr(result, "delay", None)
            return result if delay is None else delay(other.time_step)
        return super().__call__(other)


Arrow.generator_factory = Arrow.box_factory = Box
Arrow.functor_factory = Functor
Arrow.wait_factory = Wait
Id = Arrow.id
