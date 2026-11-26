# -*- coding: utf-8 -*-

"""
Augmentation and symmetrisation: the two ways to use a finite group other
than tying the weights of :mod:`discopy.neural.equivariant`.

Augmentation applies a random element of the group to both the input and the
label of a training pair, a Kleisli arrow of the state monad of seeded
randomness :data:`~discopy.kleisli.monad.Seed`. Symmetrisation averages a
function over the group, the `Reynolds operator
<https://en.wikipedia.org/wiki/Reynolds_operator>`_
:math:`R(f)(v) = \\frac{1}{|G|} \\sum_g g^{-1} f(g v)`, the projection of any
function onto the equivariant ones. A function is a homomorphism of
:math:`G`-sets exactly when it is a homomorphism of algebras of the
:class:`~discopy.monad.algebra.Writer` monad of the group.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Augment

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        sample
        augment
        augmentation
        symmetrise
        homomorphism
        is_homomorphism
        strictness

Example
-------
The average over the rotations of a map sending the first point to itself
and the others to zero is the identity, the only circulant matrix with the
same total weight on each orbit:

>>> import torch  # doctest: +EXTRA
>>> from discopy.group import Cyclic
>>> rotations = Cyclic[3].action()
>>> first = torch.tensor([[3., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
>>> mean = symmetrise(lambda v: first @ v, rotations, rotations)
>>> mean(torch.tensor([[1.], [2.], [4.]])).flatten().tolist()
[1.0, 2.0, 4.0]
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch

from discopy import para
from discopy.group import Action, FiniteGroup
from discopy.kleisli.channel import Channel
from discopy.kleisli.monad import Seed
from discopy.monad.algebra import Homomorphism, Writer
from discopy.neural.equivariant import Legs, act, check, equivariance
from discopy.neural.network import Box, Dim, Dims
from discopy.neural.torch import Module

MAX_ORDER = 10_000
""" The largest group :func:`symmetrise` sums over exactly. """


def sample(group: type[FiniteGroup]) -> Callable[[int], tuple]:
    """
    A uniform element of a group, as a computation in the
    :data:`~discopy.kleisli.monad.Seed` monad: a pure function from a seed to
    an element and the seed for the next one, as
    :func:`~discopy.kleisli.monad.uniform` does, of 63 bits, so that it fits
    the int64 key of :class:`Augment`.

    Parameters:
        group : The finite group.

    Example
    -------
    >>> from discopy.group import Cyclic
    >>> draw = sample(Cyclic[3])
    >>> g, after = draw(420)
    >>> assert draw(420) == (g, after) and after != 420
    >>> sorted({draw(seed)[0].inside for seed in range(20)})
    [(0,), (1,), (2,)]
    """
    def inside(seed: int) -> tuple:
        generator = random.Random(seed)
        return group.sample(generator), generator.getrandbits(63)
    return inside


def augment(dom: Action, cod: Action) -> Channel:
    """
    The augmentation of a pair ``(x, y)`` of an input and a label, a Kleisli
    arrow of :data:`~discopy.kleisli.monad.Seed`: the same random element
    ``g`` acts on ``x`` through ``dom`` and on ``y`` through ``cod``, both by
    :func:`~discopy.neural.equivariant.act`, so that ``y = f(x)`` implies
    ``g.y = f(g.x)`` when ``f`` is equivariant: augmentation preserves the
    relation that the equivariant functions respect.

    Parameters:
        dom : The action on the points of the inputs.
        cod : The action on the points of the labels.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> v = torch.tensor([[1.], [2.], [4.]])
    >>> (x, y), after = augment(rotations, rotations)((v, v))(420)
    >>> assert x.equal(y) and sorted(x.flatten().tolist()) == [1., 2., 4.]
    """
    draw = sample(check(dom, cod))

    def inside(pair: tuple) -> Callable[[int], tuple]:
        x, y = pair

        def computation(seed: int) -> tuple:
            g, after = draw(seed)
            return (act(dom(g), x), act(cod(g), y)), after
        return computation
    return Channel[Seed](  # ty: ignore[invalid-type-form]
        inside, tuple, tuple)


class Augment(torch.nn.Module):
    """
    The module of :func:`augmentation`: it takes a key, an input and a label
    on the legs of a network, i.e. the :func:`augment` arrow with the unit
    dimensions dropped, see :class:`~discopy.neural.equivariant.Legs`, and
    returns the key of the next draw with the augmented pair.

    The key is one non-negative integer, a seed, in a tensor with one
    element. One element ``g`` is drawn per call and acts on every row of a
    batch alike. The next key is the next seed of :func:`sample`: the box and
    :func:`augment` agree on every draw.

    Parameters:
        dom : The action on the points of the inputs.
        cod : The action on the points of the labels.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> v = torch.tensor([1., 2., 4.])
    >>> key, x, y = Augment(rotations, rotations)(torch.tensor(420), v, v)
    >>> assert x.equal(y) and key.item() != 420
    """
    def __init__(self, dom: Action, cod: Action):
        super().__init__()
        self.dom, self.cod, self.arrow = dom, cod, augment(dom, cod)

    def forward(self, key: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """
        Augment a pair with the element drawn from a key.

        Parameters:
            key : The seed, non-negative, with one element.
            x : The input, of shape ``(..., |X|)``.
            y : The label, of shape ``(..., |Y|)``.

        Returns:
            The next key, with the augmented input and label.

        Raises:
            ValueError : If the key has not one element or is negative.
        """
        if key.numel() != 1:
            raise ValueError(
                f"Expected a key with one element, "
                f"got shape {tuple(key.shape)}.")
        if int(key) < 0:
            raise ValueError(f"Expected a non-negative key, got {int(key)}.")
        pair = (Legs.restore(x, self.dom.size, 1),
                Legs.restore(y, self.cod.size, 1))
        (u, v), seed = self.arrow(pair)(int(key))
        return (key.new_full(key.shape, seed),
                u.reshape(x.shape), v.reshape(y.shape))


def augmentation(dom: Action, cod: Action, name: str = "augment") -> Box:
    """
    The box of a network augmenting a pair, with a leg for the key, one for
    the input and one for the label: the key is a scalar, of type
    ``Dims(1)``, and the points are the legs of
    :class:`~discopy.neural.equivariant.Legs`.

    Parameters:
        dom : The action on the points of the inputs.
        cod : The action on the points of the labels.
        name : The name of the box.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> box = augmentation(rotations, rotations)
    >>> assert box.dom == box.cod == Dims(1) @ Dims(3) @ Dims(3)
    >>> v = torch.tensor([1., 2., 4.])
    >>> key, x, y = box.forward(torch.tensor(420), v, v)
    >>> assert x.equal(y)
    """
    legs = Dims(1) @ Dims(Dim(dom.size, 1)) @ Dims(Dim(cod.size, 1))
    return Box(name, legs, legs, module=Augment(dom, cod))


def symmetrise(function: Callable[[torch.Tensor], torch.Tensor],
               dom: Action, cod: Action,
               samples: int | None = None, seed: int = 0
               ) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    The Reynolds operator of a function, i.e. the average of
    ``g^-1.f(g.v)`` over the group, where ``g`` acts through ``dom`` on the
    input and through ``cod`` on the output, see
    :func:`~discopy.neural.equivariant.act`. It is equivariant, it is the
    function itself if that is equivariant, and applying it twice is
    applying it once. On a linear map ``W`` it is the matrix whose entry at
    a pair of points is the mean of ``W`` on the orbit of that pair, i.e. the
    :class:`~discopy.neural.equivariant.Dense` layer with that weight on each
    orbit, see :func:`~discopy.neural.equivariant.tying`.

    The average is over the elements of the group, or over ``samples``
    elements drawn by :func:`sample` from the seed, each from the seed that
    the previous draw returned, which is an unbiased estimate of it. The
    elements are drawn once, so the result is a pure function.

    Parameters:
        function : A function of one tensor of shape ``(..., |X|, C)``.
        dom : The action on the input points.
        cod : The action on the output points.
        samples : The number of elements to average over, all if ``None``.
        seed : The seed of the draws.

    Raises:
        TypeError : If the actions are of different groups.
        ValueError : If the group is too large to sum over, or if there are
            no samples.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic, Symmetric
    >>> rotations = Cyclic[3].action()
    >>> shift = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    >>> v = torch.tensor([[1.], [2.], [4.]])
    >>> symmetrise(lambda v: shift @ v, rotations, rotations)(v).tolist()
    [[4.0], [1.0], [2.0]]
    >>> big = Symmetric[9].action()
    >>> symmetrise(lambda v: v, big, big)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ...samples...
    >>> estimate = symmetrise(lambda v: shift @ v, rotations, rotations, 5, 0)
    >>> assert estimate(v).equal(estimate(v))
    """
    group = check(dom, cod)
    if samples is None:
        if group.order() > MAX_ORDER:
            raise ValueError(
                f"{group} has {group.order()} elements, more than {MAX_ORDER}"
                ": average over samples instead.")
        elements = tuple(group.elements())
    elif samples < 1:
        raise ValueError(f"Expected at least one sample, got {samples}.")
    else:
        draw, elements, current = sample(group), [], seed
        for _ in range(samples):
            g, current = draw(current)
            elements.append(g)
    images = tuple((dom(g), cod(g.dagger())) for g in elements)

    def reynolds(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            act(out, function(act(inside, v))) for inside, out in images
        ]).mean(0)
    return reynolds


def homomorphism(function: Callable[[int], int], dom: Action, cod: Action
                 ) -> Homomorphism:
    """
    A function between the points of two actions as a homomorphism of
    algebras of the :class:`~discopy.monad.algebra.Writer` monad of the
    group: whether it commutes with the group is
    :meth:`~discopy.monad.algebra.Homomorphism.preservation` at each pair of
    an element and a point.

    Parameters:
        function : A function from the points of ``dom`` to those of ``cod``.
        dom : The action on the input points.
        cod : The action on the output points.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    The first coordinate of a pair of points commutes with the rotations:

    >>> from discopy.group import Cyclic
    >>> rotations = Cyclic[3].action()
    >>> first = homomorphism(lambda p: p // 3, rotations.tensor(rotations),
    ...                      rotations)
    >>> assert first.preservation((Cyclic[3](2), 4))
    """
    writer = Writer(check(dom, cod))
    return Homomorphism[writer](  # ty: ignore[invalid-type-form]
        function, writer.from_action(dom), writer.from_action(cod))


def is_homomorphism(function: Callable[[int], int],
                    dom: Action, cod: Action) -> bool:
    """
    Whether a function between the points of two actions commutes with the
    group, i.e. :func:`homomorphism` preserves the structure at every point
    and every generator. The generators suffice, since their images satisfy
    the relations of the group, see :class:`~discopy.group.Action`.

    Parameters:
        function : A function from the points of ``dom`` to those of ``cod``.
        dom : The action on the input points.
        cod : The action on the output points.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    The identity is a homomorphism from the rotations to themselves but not
    to the trivial action, which sends each point to its orbit:

    >>> from discopy.group import Action, Cyclic
    >>> rotations = Cyclic[3].action()
    >>> is_homomorphism(lambda x: x, rotations, rotations)
    True
    >>> is_homomorphism(lambda x: x, rotations, Action.trivial(Cyclic[3], 3))
    False
    >>> is_homomorphism(lambda x: 0, rotations, Action.trivial(Cyclic[3]))
    True
    """
    hom = homomorphism(function, dom, cod)
    return all(
        bool(hom.preservation((g, x)))
        for g in dom.group.generators() for x in range(dom.size))


def strictness(layer: para.Symmetric, dom: Action, cod: Action,
               draws: int = 20, seed: int = 0) -> float:
    """
    How far a parametric map is from a strict homomorphism: the largest
    ``|f(g.v, p) - g.f(v, p)|`` over the generators ``g``, ``draws`` random
    values ``p`` of the weights and random inputs ``v``.

    The draws can only refute strictness, so a value near zero means that no
    draw found a counterexample. The weights are one standard normal tensor
    in ``float64`` per leg of the parameter space, from a generator seeded
    by ``seed``.

    Parameters:
        layer : A parametric network, e.g. from
            :func:`~discopy.neural.equivariant.layer` and composites of them.
        dom : The action on the points of the input.
        cod : The action on the points of the output.
        draws : The number of random values of the weights.
        seed : The seed of the weights and the inputs.

    Raises:
        TypeError : If the actions are of different groups.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.group import Cyclic
    >>> from discopy.neural.equivariant import layer
    >>> from discopy.neural.network import Network
    >>> rotations = Cyclic[3].action()
    >>> strictness(layer(rotations, rotations), rotations, rotations) < 1e-9
    True

    A dense box with all nine weights free is not equivariant:

    >>> x, w = Dims(3), Dims(Dim(3, 3))
    >>> dense = para.Markov[Network](x, x, param=w, inside=Box(
    ...     "W", x @ w, x, module=lambda v, W: v @ W))
    >>> strictness(dense, rotations, rotations) > 1e-2
    True
    """
    check(dom, cod)
    generator = torch.Generator().manual_seed(seed)
    (leg, ) = layer.dom.inside

    def draw(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    def distance(module: Module) -> float:
        def function(u: torch.Tensor) -> torch.Tensor:
            out = module(u.reshape(*u.shape[:-2], *leg.inside))
            return out.reshape(*u.shape[:-2], cod.size, -1)
        v = draw((4, ) + leg.inside).reshape(4, dom.size, -1)
        return equivariance(function, dom, cod, v)
    shapes = [dim.inside for dim in layer.param.inside]
    return max((
        distance(Module(layer, *map(draw, shapes))) for _ in range(draws)),
        default=0.0)
