# -*- coding: utf-8 -*-

"""
The traced category of feedforward neural networks: lists of shapes as
objects, layers as boxes, with copy, discard, merge, noise, swap and trace.

A :class:`Dims` is a list of :class:`Dim`, i.e. a tuple of tensors with one
shape per leg. A :class:`Box` from ``Dims(a_1, ..., a_m)`` to
``Dims(b_1, ..., b_n)`` is a layer taking ``m`` tensors and returning ``n``,
computed by its optional :attr:`Box.module`. A :class:`Network` is a diagram
of such layers: composition ``>>`` feeds the outputs of one into the inputs
of the next, tensor ``@`` runs two side by side, :meth:`Network.copy` sends
an activation to several layers, :meth:`Network.discard` drops one and
:meth:`Network.trace` feeds an output back into an input, as a recurrent
network does. Networks form the free traced Markov and comarkov category on
their boxes, see :mod:`discopy.markov`, :mod:`discopy.comarkov` and
:mod:`discopy.feedback`. A feedforward network
compiles to a function on tensors with :meth:`Network.to_function`, which
:mod:`discopy.neural.torch` wraps as a PyTorch module and ``jax.jit`` takes
as it is, and to its reverse derivative with :meth:`Network.to_lens`, a
:class:`Lens` whose ``put`` is backpropagation.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dims
    Network
    Box
    Function
    Cofunction
    Swap
    Permutation
    Copy
    Merge
    Discard
    Noise
    Feedback
    Trace
    FollowedBy
    Head
    Tail
    Functor
    Lens
    Learner

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        autodiff

Example
-------

A residual block copies its input, applies a layer to one copy and adds the
two back together; a recurrent cell feeds one of its outputs back in.

>>> x, h = Dims(4), Dims(8)
>>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
>>> residual = Network.copy(x) >> layer @ x >> add
>>> assert residual.dom == residual.cod == x
>>> cell = Box('cell', x @ h, x @ h)
>>> recurrent = cell.trace()
>>> assert recurrent.dom == recurrent.cod == x
>>> Equation(residual, recurrent, symbol='').draw(
...     doctest="docs/_static/neural/residual-and-recurrent.svg")

.. image:: /_static/neural/residual-and-recurrent.svg
    :align: center
"""

from __future__ import annotations

from typing import Self

from discopy import (
    comarkov, feedback, lens, markov, messages, monoidal, para, python,
    testable)
from discopy.abc import MarkovCategory
from discopy.cat import factory
from discopy.monoidal import Dim
from discopy.testable import axiom
from discopy.utils import (
    AxiomError, assert_isinstance, factory_name, tuplify, untuplify)


@factory
class Dims(monoidal.Ty):
    """
    A list of shapes, i.e. the free monoid over :class:`Dim` with
    concatenation as tensor ``@`` and the empty list as unit: a tuple of
    tensors, one per leg.

    Together with the product ``*`` of shapes, which distributes over the
    concatenation, this is the free rig on the natural numbers: ``Dims()``
    is the zero and ``Dims(1)`` the one, both operations commutative only
    up to :class:`Swap`. Its laws are axioms: :meth:`product_unitality`,
    :meth:`product_associativity`, :meth:`absorption` and
    :meth:`distributivity_right` hold on the nose,
    :meth:`distributivity_left` up to a permutation of the legs.

    Parameters:
        inside : The shape of each leg, an integer standing for a vector.

    Example
    -------
    >>> assert Dims(2, 3) == Dims(Dim(2), Dim(3)) == Dims(2) @ Dims(3)
    >>> assert Dims(Dim(2, 3)) != Dims(2, 3) and Dims(1) == Dims(Dim())
    >>> Dims(2, 3) * Dims(4)
    Dims(Dim(2, 4), Dim(3, 4))
    >>> from discopy.utils import dumps, loads
    >>> assert loads(dumps(Dims(2, Dim(2, 3)))) == Dims(2, Dim(2, 3))
    """
    generator_factory = Dim
    box_factory = None
    """ No generator of one colour is named, so there is no canonical
    instance of a law, see :meth:`discopy.testable.Axiom.canonical`. """

    def __init__(self, *inside: int | Dim, **kwargs):
        inside = kwargs.pop('inside', inside)
        super().__init__(
            *(Dim(x) if isinstance(x, int) else x for x in inside), **kwargs)

    def __mul__(self, other: Dims) -> Dims:
        assert_isinstance(other, Dims)
        return type(self)(*(x @ y for x in self.inside for y in other.inside))

    @axiom
    def product_unitality(cls, x: Self) -> testable.Equation[Self]:
        """ ``Dims(1)`` is the unit of the product, on both sides. """
        one = type(x)(1)
        return testable.Equation(one * x, x, x * one)

    @axiom
    def product_associativity(
            cls, x: Self, y: Self, z: Self) -> testable.Equation[Self]:
        """ The product is associative. """
        return testable.Equation((x * y) * z, x * (y * z))

    @axiom
    def absorption(cls, x: Self) -> testable.Equation[Self]:
        """ ``Dims()`` absorbs the product, on both sides. """
        zero = type(x)()
        return testable.Equation(zero * x, zero, x * zero)

    @axiom
    def distributivity_right(
            cls, x: Self, y: Self, z: Self) -> testable.Equation[Self]:
        """ The product distributes over a concatenation on its left. """
        return testable.Equation((x @ y) * z, (x * z) @ (y * z))

    @axiom
    def distributivity_left(
            cls, x: Self, y: Self, z: Self) -> testable.Equation[Self]:
        """
        The product distributes over a concatenation on its right, up to
        a permutation of the legs: ``x * (y @ z)`` lists the legs of ``y``
        and ``z`` for each leg of ``x`` in turn, ``(x * y) @ (x * z)``
        those of ``y`` for every leg of ``x`` first. That permutation is the
        left distributor of a rig category :cite:p:`Laplaza72`, where the
        right one is an identity.
        """
        return testable.Equation(
            x * (y @ z), (x * y) @ (x * z), up_to=cls.legs)

    def legs(self) -> list[tuple[int, ...]]:
        """
        The shapes of the legs in lexicographic order, i.e. the list up to
        permutation.

        >>> Dims(3, Dim(2, 4), 2).legs()
        [(2,), (2, 4), (3,)]
        """
        return sorted(dim.inside for dim in self.inside)

    def __repr__(self):
        return f"Dims({', '.join(map(repr, self.inside))})"

    __str__ = __repr__

    delay, head, tail = Dim.delay, Dim.head, Dim.tail

    def check(self, *tensors) -> None:
        """
        Raise ``AxiomError`` unless there is one tensor per leg with the
        shape of the leg as its trailing dimensions, whatever comes first.

        Parameters:
            tensors : One array per leg, with a ``shape``.

        Example
        -------
        >>> import numpy as np
        >>> Dims(2, Dim(2, 3)).check(np.ones((5, 2)), np.ones((5, 2, 3)))
        >>> Dims(2).check(np.ones((5, 3)))
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: Expected a shape ending in (2,), got (5, 3).
        """
        if len(tensors) != len(self):
            raise AxiomError(
                f"Expected {len(self)} tensors, got {len(tensors)}.")
        for tensor, dim in zip(tensors, self.inside):
            shape = tuple(tensor.shape)
            if shape[len(shape) - len(dim):] != dim.inside:
                raise AxiomError(
                    f"Expected a shape ending in {dim.inside}, got {shape}.")


@factory
class Network(markov.Diagram, comarkov.Diagram):
    """
    A network is a Markov and comarkov diagram with lists of shapes as
    objects and :class:`Box` layers inside, together with the trace.

    Parameters:
        inside (Layer) : The layers of the network.
        dom (Dims) : The input legs of the network.
        cod (Dims) : The output legs of the network.

    Note
    ----
    Networks can be written as Python functions of their input legs.

    >>> x = Dims(4)
    >>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
    >>> @Network.from_callable(x, x)
    ... def residual(v):
    ...     return add(layer(v), v)
    >>> assert residual == Network.copy(x) >> layer @ x >> add
    """
    ob = Dims

    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        """
        The spiders of a network are its copies and merges, with one leg in
        or one leg out: there is no Frobenius law between the two.
        """
        if phase is not None or 1 not in (n_legs_in, n_legs_out):
            raise ValueError(messages.NO_SUCH_SPIDER.format(
                factory_name(cls), "copies and merges"))
        return cls.copy_factory(typ, n_legs_out) if n_legs_in == 1\
            else cls.merge_factory(typ, n_legs_in)

    def to_function(self, tensor: type = object) -> python.Function:
        """
        Compile a feedforward network to a function on tensors, one per leg:
        the functor sending every leg to ``tensor`` and every box to its
        :meth:`Box.forward`, with the copy, discard and swap of
        :class:`python.Function`. The function is what ``jax.jit`` compiles
        and :class:`discopy.neural.torch.Module` runs.

        Parameters:
            tensor : The type of the tensors on the legs, e.g. ``jax.Array``
                or ``torch.Tensor``, checked on every wire.

        Example
        -------
        >>> import numpy as np
        >>> x = Dims(2)
        >>> double = Box('double', x, x, module=lambda v: 2 * v)
        >>> add = Box('add', x @ x, x, module=lambda v, w: v + w)
        >>> residual = Network.copy(x) >> double @ x >> add
        >>> residual.to_function(np.ndarray)(np.array([1, 2]))
        array([3, 6])
        >>> (add >> double).trace().to_function()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        NotImplementedError: Trace(add >> double) has a trace, only ...
        """
        self.assert_isfeedforward()
        return Functor(ob_map=lambda leg: (tensor, ),
                       ar_map=lambda box: box.assert_hasmodule().forward,
                       cod=python.Function)(self)

    def assert_isfeedforward(self) -> None:
        """ Raise ``NotImplementedError`` on a trace or a dagger: neither
        compiles to a function on tensors. """
        if any(isinstance(box, Trace) for box in self.boxes):
            raise NotImplementedError(
                f"{self} has a trace, only feedforward networks compile.")
        for box in self.boxes:
            if box.is_dagger:
                raise NotImplementedError(f"{box} has no adjoint module.")

    def to_lens(self, tensor: type = object) -> Lens:
        """
        The reverse derivative of a feedforward network
        :cite:p:`CruttwellEtAl22`: the lens whose ``get`` is
        :meth:`to_function` and whose ``put`` takes one tensor per input
        leg, then one gradient per output leg, to one gradient per input
        leg. It is the functor into :class:`Lens` sending every leg to a
        pair of ``tensor`` and every box to its :meth:`Box.forward` and
        :meth:`Box.backward`: composition is the chain rule, a copy sums
        the gradients of its branches and a discard gives zero.

        Parameters:
            tensor : The type of the tensors and of their gradients.

        Example
        -------
        >>> import numpy as np
        >>> x = Dims(2)
        >>> square = Box('square', x, x, module=lambda v: v * v,
        ...              vjp=lambda v, dv: 2 * v * dv)
        >>> add = Box('add', x @ x, x, module=lambda v, w: v + w,
        ...           vjp=lambda v, w, dv: (dv, dv))
        >>> residual = Network.copy(x) >> square @ x >> add
        >>> backprop = residual.to_lens(np.ndarray)
        >>> backprop.get(np.array([1, 2]))
        array([2, 6])
        >>> backprop.put(np.array([1, 2]), np.array([1, 1]))
        array([3, 5])
        """
        self.assert_isfeedforward()
        leg = Lens.ob((tensor, ), (tensor, ))

        def reverse(box):
            box.assert_hasmodule()
            dom, cod = (leg ** len(legs) for legs in (box.dom, box.cod))
            return Lens(
                dom, cod,
                python.Function(box.forward, dom.positive, cod.positive),
                python.Function(box.backward, dom.positive @ cod.negative,
                                dom.negative))

        return Functor(ob_map=lambda _: leg, ar_map=reverse, cod=Lens)(self)


class Box(markov.Box, Network):
    """
    A box is a layer taking one tensor per leg of ``dom`` and returning one
    per leg of ``cod``, computed by its ``module`` when it has one.

    Parameters:
        name : The name of the layer.
        dom : The input legs of the layer.
        cod : The output legs of the layer.
        module : The object computing the layer, e.g. a ``torch.nn.Module``.
        vjp : The vector-Jacobian product of the layer, from one tensor per
            input leg then one gradient per output leg to one gradient per
            input leg; :func:`autodiff` of the module by default.

    Note
    ----
    The module is the ``data`` of the box, so boxes compare equal when they
    have the same name, legs and module, a framework's modules comparing by
    identity. The repr and the serialisation omit the module, which has no
    eval-able representation, so ``eval(repr(f)) == f`` and
    ``loads(dumps(f)) == f`` hold for a box without one and give the shape
    of one with. The ``vjp`` is how the derivative of the module is
    computed, not part of what the box is: it is kept by :meth:`rebuild`
    and read by neither the equality, the repr nor the serialisation.

    Example
    -------
    >>> f = Box('f', Dims(2), Dims(3, 3), module=lambda x: (x, x))
    >>> assert f.module is f.data is f.dagger().module
    >>> f
    neural.network.Box('f', Dims(Dim(2)), Dims(Dim(3), Dim(3)))
    """
    module = vjp = None
    serialised_attrs = (*markov.Box.serialised_attrs, "vjp")

    def __init__(self, name: str, dom: Dims, cod: Dims,
                 module: object = None, data=None, vjp=None, **params):
        self.module, self.vjp = data if module is None else module, vjp
        super().__init__(name, dom, cod, data=self.module, **params)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return f"{factory_name(type(self))}("\
            f"{self.name!r}, {self.dom!r}, {self.cod!r})"

    def to_tree(self) -> dict:
        tree = super().to_tree()
        tree.pop('data', None)
        return tree

    def forward(self, *tensors):
        """
        Apply the module to one tensor per leg of ``dom`` and return one per
        leg of ``cod``, each checked against its leg by :meth:`Dims.check`.

        Parameters:
            tensors : One tensor per input leg.

        Example
        -------
        >>> import numpy as np
        >>> split = Box('split', Dims(4), Dims(2, 2),
        ...             module=lambda v: (v[:2], v[2:]))
        >>> split.forward(np.arange(4))
        (array([0, 1]), array([2, 3]))
        >>> Box('f', Dims(4), Dims(2), module=lambda v: v).forward(np.ones(4))
        Traceback (most recent call last):
            ...
        discopy.utils.AxiomError: f: Expected a shape ending in (2,), got (4,).
        """
        self.assert_hasmodule()
        try:
            self.dom.check(*tensors)
            outputs = tuplify(
                self.module(*tensors))  # ty: ignore[call-non-callable]
            self.cod.check(*outputs)
        except AxiomError as error:
            raise AxiomError(f"{self}: {error}") from error
        return untuplify(outputs)

    def assert_hasmodule(self) -> Box:
        """ Raise ``ValueError`` unless the box has a module, else return
        the box itself. """
        if self.module is None:
            raise ValueError(f"{self} has no module.")
        return self

    def backward(self, *tensors):
        """
        Apply the vector-Jacobian product of the layer to one tensor per
        leg of ``dom`` then one gradient per leg of ``cod``, and return one
        gradient per leg of ``dom``, each checked against its leg: the
        :attr:`vjp` of the box when it has one, else :func:`autodiff` of
        its module.

        Parameters:
            tensors : One tensor per input leg, one gradient per output leg.

        Example
        -------
        >>> import numpy as np
        >>> scale = Box('scale', Dims(2), Dims(2), module=lambda v: 3 * v,
        ...             vjp=lambda v, dv: 3 * dv)
        >>> scale.backward(np.array([1, 2]), np.array([1, 0]))
        array([3, 0])
        """
        self.assert_hasmodule()
        inputs, gradients = tensors[:len(self.dom)], tensors[len(self.dom):]
        try:
            self.dom.check(*inputs)
            self.cod.check(*gradients)
            outputs = tuplify(
                autodiff(self.module, inputs, gradients) if self.vjp is None
                else self.vjp(*tensors))
            self.dom.check(*outputs)
        except AxiomError as error:
            raise AxiomError(f"{self}: {error}") from error
        return untuplify(outputs)


class Function(markov.Function, Box):
    """
    The copies, discards and permutations of the legs of a network.

    Parameters:
        dom (Dims) : The legs to copy, discard and permute.
        func : The leg each output reads.
    """


class Permutation(markov.Permutation, Box):
    """
    A permutation of the legs of a network.

    Parameters:
        dom (Dims) : The legs to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """


class Cofunction(comarkov.Cofunction, Box):
    """
    The permutations, merges and noise of the legs of a network.

    Parameters:
        cod (Dims) : The legs to merge into.
        cofunc : The leg each input goes to.
    """


Function.register(Permutation)
Cofunction.register(Permutation)


class Swap(Permutation, markov.Swap, Box):
    """
    The swap of two legs.

    Parameters:
        left (Dims) : The leg on the top left and bottom right.
        right (Dims) : The leg on the top right and bottom left.
    """


class Copy(markov.Copy, Function):
    """
    The copy of a leg some ``n`` number of times.

    Parameters:
        x (Dims) : The leg to copy.
        n : The number of copies.
    """


class Merge(comarkov.Merge, Cofunction):
    """
    The merge of ``n`` copies of a leg, the dagger of :class:`Copy`.

    Parameters:
        x (Dims) : The leg to merge.
        n : The number of copies.
    """


class Discard(markov.Discard, Copy):
    """
    The discard of a leg.

    Parameters:
        x (Dims) : The leg to discard.
    """


class Noise(comarkov.Noise, Merge):
    """
    The noise on a leg, the dagger of :class:`Discard`.

    Parameters:
        x (Dims) : The leg of the noise.
    """


class Functor(markov.Functor, comarkov.Functor):
    """
    A functor from networks to any traced category with the structure the
    network uses, e.g. an evaluation of the layers as functions.

    Parameters:
        ob_map (Mapping[Dims, Dims]) :
            Map from atomic :class:`Dims` to :code:`cod.ob`.
        ar_map (Mapping[Box, Network]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Network` by default.

    Example
    -------
    >>> from discopy import python
    >>> x = Dims(2)
    >>> layer, add = Box('layer', x, x), Box('add', x @ x, x)
    >>> residual = Network.copy(x) >> layer @ x >> add
    >>> F = Functor(
    ...     ob_map={x: (list, )},
    ...     ar_map={layer: lambda v: [2 * i for i in v],
    ...             add: lambda v, w: [i + j for i, j in zip(v, w)]},
    ...     cod=python.Function)
    >>> assert F(residual)([1, 2]) == [3, 6]
    """
    dom = cod = Network


class Lens(lens.Markov, MarkovCategory):
    """
    The lenses of functions on tensors, where the reverse derivative of a
    network lives, see :meth:`Network.to_lens`: ``get`` runs the network
    forward, ``put`` takes its inputs and the gradients of its outputs to
    the gradients of its inputs. They form a Markov category as those of
    :class:`discopy.lens.Comarkov` do, with the sum of the gradients as
    merge and zero as its unit.

    Parameters:
        dom (lens.Ty) : The types of the inputs and of their gradients.
        cod (lens.Ty) : The types of the outputs and of their gradients.
        get (python.Function) : The forward pass.
        put (python.Function) : The backward pass.

    Example
    -------
    >>> x = Lens.ob((float, ), (float, ))
    >>> Function = python.Function
    >>> square = Lens(
    ...     x, x, Function(lambda a: a * a, (float, ), (float, )),
    ...     Function(lambda a, da: 2 * a * da, (float, float), (float, )))
    >>> fan_out = Lens.copy(x) >> square @ square
    >>> fan_out.get(3.)
    (9.0, 9.0)
    >>> fan_out.put(3., 1., 1.)
    12.0
    >>> (Lens.copy(x) >> square @ Lens.discard(x)).put(3., 1.)
    6.0

    Note
    ----
    The ``put`` of a composition runs the ``get`` of what comes first again,
    see :meth:`discopy.lens.Markov.then`, so that the backward pass of a
    network of depth ``n`` runs its layers forward a number of times
    quadratic in ``n``, see `next#516
    <https://github.com/rel-int/discopy-next/issues/516>`_.
    """
    category = python.Function

    @classmethod
    def checked_params(cls) -> tuple:
        """ No subscript is checked: functions are not generated. """
        return ()

    @classmethod
    def copy(cls, x: lens.Ty, n: int = 2) -> Lens:
        """
        The copy of a pair ``n`` times: ``get`` copies the tensors, ``put``
        sums the ``n`` gradients of each, zero when there is none.

        Parameters:
            x : The pair of types to copy.
            n : The number of copies.
        """
        width = len(x.positive)

        def put(*tensors):
            inputs, gradients = tensors[:width], tensors[width:]
            return untuplify(tuple(
                sum(gradients[i::width], 0 * tensor)
                for i, tensor in enumerate(inputs)))

        return cls(x, x ** n, cls.category.copy(x.positive, n), cls.category(
            put, x.positive @ (x ** n).negative, x.negative))

    @classmethod
    def descent(cls, x: lens.Ty, rate: float) -> Lens:
        """
        Gradient descent at a learning ``rate``, the lens into a pair ``x``
        of weights and gradients from the pair of the weights and their
        new values: ``get`` is the identity, ``put`` subtracts ``rate``
        times its gradient from each weight :cite:p:`CruttwellEtAl22`.

        Parameters:
            x : The pair of the types of the weights and of their gradients.
            rate : The learning rate.

        Example
        -------
        >>> x = Lens.ob((float, ), (float, ))
        >>> assert Lens.descent(x, rate=.5).get(3.) == 3.
        >>> assert Lens.descent(x, rate=.5).put(3., 2.) == 2.
        """
        weights = x.positive

        def put(*tensors):
            return untuplify(tuple(
                weight - rate * gradient for weight, gradient in zip(
                    tensors[:len(weights)], tensors[len(weights):])))

        return cls(cls.ob(weights, weights), x, cls.category.id(weights),
                   cls.category(put, weights @ x.negative, weights))


class Learner(para.Markov):
    """
    A learner is a parametric lens of functions on tensors
    :cite:p:`FongEtAl19,CruttwellEtAl22`, i.e. a parametric map over
    :class:`Lens`: its ``inside`` is a lens from the inputs and the weights
    to the outputs, its ``param`` the pair of the types of the weights and
    of what ``put`` returns for them, their gradients.

    Composition tensors the weights, :meth:`reparam` by :meth:`Lens.copy`
    ties them, so that ``put`` sums their gradients, and by
    :meth:`Lens.descent` turns the gradients into the new weights:
    :meth:`descent` is one step of an optimiser.

    Parameters:
        dom (lens.Ty) : The types of the inputs and of their gradients.
        cod (lens.Ty) : The types of the outputs and of their gradients.
        inside (Lens) : The lens ``dom @ param -> cod``.
        param (lens.Ty) : The types of the weights and of their gradients.

    Example
    -------
    >>> import numpy as np
    >>> x = Dims(2)
    >>> scale = para.Markov[Network](x, x, param=x, inside=Box(
    ...     'scale', x @ x, x, module=lambda v, w: v * w,
    ...     vjp=lambda v, w, dv: (w * dv, v * dv)))
    >>> layer = Learner.reverse(scale, np.ndarray)
    >>> learner = layer >> layer
    >>> v, w, dv = np.array([1., 2.]), np.array([3., 4.]), np.ones(2)
    >>> learner.inside.get(v, w, w)
    array([ 9., 32.])
    >>> learner.inside.put(v, w, w, dv)
    (array([ 9., 16.]), array([3., 8.]), array([3., 8.]))
    >>> tied = learner.reparam(Lens.copy(layer.param))
    >>> tied.inside.put(v, w, dv)
    (array([ 9., 16.]), array([ 6., 16.]))
    >>> tied.descent(rate=.5).inside.put(v, w, dv)
    (array([ 9., 16.]), array([ 0., -4.]))
    """
    category = Lens

    @classmethod
    def checked_params(cls) -> tuple:
        """ No subscript is checked: functions are not generated. """
        return ()

    @classmethod
    def reverse(cls, network: para.Symmetric, tensor: type = object
                ) -> Learner:
        """
        The reverse derivative of a parametric network, i.e. that of its
        inside by :meth:`Network.to_lens`, a pair of ``tensor`` for each of
        its legs, weights included.

        Parameters:
            network : The parametric network, over :class:`Network`.
            tensor : The type of the tensors and of their gradients.
        """
        leg = Lens.ob((tensor, ), (tensor, ))
        dom, cod, param, copar = (leg ** len(legs) for legs in (
            network.dom, network.cod, network.param, network.copar))
        return cls(dom, cod, network.inside.to_lens(tensor), param, copar)

    def descent(self, rate: float) -> para.Symmetric:
        """
        One step of gradient descent at a learning ``rate``: the learner
        whose ``put`` returns the new weights where that of ``self`` returns
        their gradients, i.e. :meth:`reparam` by :meth:`Lens.descent`.

        Parameters:
            rate : The learning rate.
        """
        step = Lens.descent(self.param, rate)
        return self.reparam(step)  # ty: ignore[invalid-argument-type]


def autodiff(module, inputs: tuple, gradients: tuple) -> tuple:
    """
    The vector-Jacobian product of a ``module`` at its ``inputs``, applied
    to the ``gradients`` of its outputs, by the reverse mode of the
    framework the tensors come from: PyTorch or JAX.

    Parameters:
        module : The function to differentiate, from tensors to tensors.
        inputs : The tensors to differentiate at, one per argument.
        gradients : The gradients of the outputs, one per output.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> x, dy = torch.tensor([1., 2.]), torch.tensor([1., 1.])
    >>> autodiff(lambda v: v * v, (x, ), (dy, ))
    (tensor([2., 4.]),)
    """
    if not inputs:
        return ()
    framework = type(inputs[0]).__module__.partition(".")[0]
    if framework == "torch":
        # pylint: disable-next=import-outside-toplevel  # no framework above
        import torch
        with torch.enable_grad():
            inputs = tuple(x.detach().requires_grad_() for x in inputs)
            live = [pair for pair in zip(tuplify(module(*inputs)), gradients)
                    if pair[0].requires_grad]
            if not live:
                return tuple(0 * x for x in inputs)
            outputs, gradients = zip(*live)
            return torch.autograd.grad(
                outputs, inputs, gradients,
                allow_unused=True, materialize_grads=True)
    if framework in ("jax", "jaxlib"):
        # pylint: disable-next=import-outside-toplevel  # no framework above
        import jax
        return jax.vjp(
            lambda *xs: tuplify(module(*xs)), *inputs)[1](tuple(gradients))
    raise NotImplementedError(
        f"No reverse mode for {framework}: give the box a vjp.")


Id = Network.id
Equation = markov.Equation

Feedback, Trace, FollowedBy, Head, Tail = feedback.structural_boxes(
    Network, Box)
Network.box_factory = Box
Network.functor_factory = Functor
Network.copy_factory, Network.merge_factory = Copy, Merge
Network.function_factory = Function
Network.cofunction_factory = Cofunction
Network.discard_factory, Network.noise_factory = Discard, Noise
Network.swap_factory = Swap
Network.permutation_factory = Permutation
