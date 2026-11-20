# -*- coding: utf-8 -*-

"""
The category of matrices with the Kronecker product as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dim
    Tensor
    Functor
    Diagram
    CMap
    Box
    Permutation
    Swap
    Cup
    Cap
    Spider
    Sum
    Bubble
    Diff
    Term
    Constant
    Variable
    Application
    Abstraction
    Tuple
    Equation

Tensor combinatorial maps
-------------------------

A :class:`CMap` is a tensor network stored as a combinatorial map, whose
boxes are tensors, edges are summed indices and boundary ports are free
indices. Swaps, cups and caps become wiring while spiders stay as boxes.

>>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
>>> assert (vector >> vector[::-1]).to_map().eval().array == 1

>>> with backend('jax'):  # doctest: +EXTRA
...     import jax, jax.numpy as jnp
...     b = lambda x: Box[float]('v', Dim(1), Dim(2), x * jnp.ones(2))
...     f = lambda x: (b(x) >> b(x)[::-1]).to_map().eval().array
...     assert jax.grad(f)(1.) == 4.
"""

from __future__ import annotations

import math
import numbers
import sys
from itertools import count
from typing import TYPE_CHECKING, Callable, ClassVar, Sequence, TypeVar

from discopy import (
    cat, monoidal, rigid, feedback, frobenius, cmap, config, hypergraph,
    messages)
from discopy.cat import factory, assert_iscomposable
from discopy.frobenius import Dim, Cup
from discopy.matrix import (  # noqa: F401  pylint: disable=unused-import
    Matrix, backend, set_backend, get_backend, repr_nonfinite,
    Backend, NumPy, JAX, PyTorch, TensorFlow, Contractor, OptEinsum, Quimb)
from discopy.testable import TestableGeneric, no_strategy
from discopy.python import finset
from discopy.utils import (
    factory_key, factory_name, assert_isinstance, product, assert_isatomic,
    assert_istraceable)

if TYPE_CHECKING:
    import sympy
    import tensornetwork
    import quimb.tensor


@factory
class Tensor(Matrix):
    """
    A tensor is a :class:`Matrix` with dimensions as domain and codomain and
    the Kronecker product as tensor.

    Parameters:
        inside : The array inside the tensor.
        dom : The domain dimension.
        cod : The codomain dimension.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            dagger
            cups
            caps
            swap
            spiders
            transpose
            conjugate
            round
            subs
            grad
            jacobian

    Examples
    --------
    >>> m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    >>> v = Tensor([0, 1], Dim(1), Dim(2))
    >>> v >> m >> v.dagger()
    Tensor[int]([0], dom=Dim(1), cod=Dim(1))

    Notes
    -----
    Tensors can have sympy symbols as free variables.

    >>> from sympy import Expr  # doctest: +EXTRA
    >>> from sympy.abc import phi, psi
    >>> v = Tensor[Expr]([phi, psi], Dim(1), Dim(2))
    >>> d = v >> v.dagger()
    >>> assert v >> v.dagger() == Tensor[Expr](
    ...     [phi * phi.conjugate() + psi * psi.conjugate()], Dim(1), Dim(1))

    These can be substituted and lambdifed.

    >>> v.subs(phi, 0).lambdify(psi, dtype=int)(1)
    Tensor[int]([0, 1], dom=Dim(1), cod=Dim(2))

    We can also use jax.numpy using :func:`backend`.

    >>> with backend('jax'):
    ...     f = lambda *xs: d.lambdify(phi, psi, dtype=float)(*xs).array
    ...     import jax
    ...     assert jax.grad(f)(1., 2.) == 2.
    """
    ob = Dim

    @classmethod
    def checked_params(cls) -> tuple:
        """
        None, the Kronecker product of the dimensions a law draws growing
        exponentially in their number, where the direct sum of a matrix
        grows linearly.
        """
        return ()

    def __init__(self, array, dom: Dim, cod: Dim):
        assert_isinstance(dom, Dim)
        assert_isinstance(cod, Dim)
        super().__init__(array, product(dom.inside), product(cod.inside))
        self.array = self.array.reshape(dom.inside + cod.inside)
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom=Dim(1)) -> Tensor:
        return cls(Matrix.id(product(dom.inside)).array, dom, dom)

    @classmethod
    def strategy(cls, *, dom=None, cod=None, max_size=3, max_entry=3,
                 max_entries=None):
        """
        Generate tensors between dimensions of at most two factors up to
        ``max_size``, with entries in ``range(max_entry + 1)`` and at most
        ``max_entries`` of them as :meth:`Matrix.strategy` does between
        natural numbers.
        """
        from hypothesis import strategies as st

        typed = cls[cls.dtype or int]
        dims = st.lists(st.integers(min_value=1, max_value=max_size),
                        max_size=2).map(lambda ns: Dim(*ns))

        @st.composite
        def tensors(draw):
            """ Draw the shape, then one entry per coefficient. """
            shape = tuple(draw(dims) if x is None else x for x in (dom, cod))
            size = product(shape[0].inside) * product(shape[1].inside)
            entries = st.lists(
                st.integers(min_value=0, max_value=max_entry),
                min_size=size, max_size=size)
            return typed(draw(entries), *shape)

        return tensors() if max_entries is None else tensors().filter(
            lambda tensor: product(tensor.dom.inside + tensor.cod.inside)
            <= max_entries)

    def then(self, *others: Tensor) -> Tensor:
        if len(others) != 1:
            return super().then(*others)
        other, = others
        assert_isinstance(other, self.ar)
        assert_iscomposable(self, other)
        cls = self.promote(other)
        left, right = (
            factor.cast(cls.dtype).array for factor in (self, other))
        with backend() as np:
            array = np.tensordot(left, right, len(self.cod))\
                if left.shape and right.shape else left * right
        return cls(array, self.dom, other.cod)

    def tensor(self, *others: Tensor) -> Tensor:
        """
        The tensor of ``n`` tensors, i.e. their outer product with the axes
        of every domain moved in front of those of every codomain.

        >>> v = Tensor([1, 0], Dim(1), Dim(2))
        >>> assert v.tensor(v, v) == v @ v @ v
        """
        if not others:
            return self
        factors = (self, ) + others
        for other in others:
            assert_isinstance(other, self.ar)
        dom = self.dom.tensor(*(other.dom for other in others))
        cod = self.cod.tensor(*(other.cod for other in others))
        target, i, j = [], 0, len(dom)
        for factor in factors:
            target += list(range(i, i + len(factor.dom)))
            target += list(range(j, j + len(factor.cod)))
            i, j = i + len(factor.dom), j + len(factor.cod)
        source = list(range(len(target)))
        cls = self.promote(*others)
        array, *arrays = (factor.cast(cls.dtype).array for factor in factors)
        with backend() as np:
            for other in arrays:
                array = np.tensordot(array, other, 0)\
                    if array.shape and other.shape else array * other
            if source:
                array = np.moveaxis(array, source, target)
        return cls(array, dom, cod)

    def dagger(self) -> Tensor:
        source = list(range(len(self.dom @ self.cod)))
        target = [i + len(self.cod) if i < len(self.dom) else
                  i - len(self.dom) for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, source, target))
        return type(self)(array, self.cod, self.dom)

    def trace(self, n=1, left=False) -> Tensor:
        """
        The trace of a tensor over its last ``n`` factors, i.e. the
        contraction of each of the last ``n`` axes of its domain with the
        matching axis of its codomain.

        Parameters:
            n : The number of factors to trace.
            left : Whether to trace the first ``n`` factors instead.

        Example
        -------
        >>> assert Tensor.swap(Dim(2), Dim(2)).trace() == Tensor.id(Dim(2))
        >>> f = Tensor[int](list(range(16)), Dim(2, 2), Dim(2, 2))
        >>> assert f.trace() == Tensor[int]([5, 9, 21, 25], Dim(2), Dim(2))
        >>> assert f.trace(left=True)\\
        ...     == Tensor[int]([10, 12, 18, 20], Dim(2), Dim(2))
        """
        assert_istraceable(self, n, left)
        a, b = len(self.dom), len(self.cod)
        dom_rest, cod_rest = (
            (slice(n, None), slice(n, None)) if left
            else (slice(0, a - n), slice(0, b - n)))
        dom_traced, cod_traced = (
            (range(n), range(a, a + n)) if left
            else (range(a - n, a), range(a + b - n, a + b)))
        labels = list(range(a + b))
        for i, j in zip(dom_traced, cod_traced):
            labels[j] = labels[i]
        output = labels[:a][dom_rest] + labels[a:][cod_rest]
        with backend() as np:
            array = np.einsum(self.array, labels, output)
        dom, cod = self.dom[dom_rest], self.cod[cod_rest]
        return type(self)(array, dom, cod)

    def repeat(self) -> Tensor:
        """
        The reflexive transitive closure of a Boolean tensor, i.e. of the
        square matrix it reshapes to, with :meth:`Matrix.repeat`.

        Example
        -------
        >>> swap = Tensor[bool].swap(Dim(2), Dim(2))
        >>> assert swap.repeat() == swap + Tensor[bool].id(Dim(2, 2))
        """
        if self.dtype != bool or self.dom != self.cod:
            raise TypeError(messages.MATRIX_REPEAT_ERROR)
        size = product(self.dom.inside)
        array = Matrix[bool](self.array.reshape(size, size), size, size)\
            .repeat().array
        return type(self)(array, self.dom, self.cod)

    @classmethod
    def cup_factory(cls, left: Dim, right: Dim) -> Tensor:
        """
        The cup of an atomic dimension, and of a compound one its
        :meth:`cups`, which pair the factors nested.

        Example
        -------
        >>> x = Dim(2, 3)
        >>> assert Tensor.cup_factory(x, x.r) == Tensor.cups(x, x.r)
        """
        assert_isinstance(left, Dim)
        assert_isinstance(right, Dim)
        if len(left) != 1:
            return cls.cups(left, right)
        left.assert_isadjoint(right)
        return cls(cls.id(left).array, left @ right, Dim(1))

    @classmethod
    def cups(cls, left: Dim, right: Dim) -> Tensor:
        return rigid.nesting(cls, cls.cup_factory)(left, right)

    @classmethod
    def caps(cls, left: Dim, right: Dim) -> Tensor:
        return cls.cups(left, right).dagger()

    @classmethod
    def swap(cls, left: Dim, right: Dim) -> Tensor:
        dom, cod = left @ right, right @ left
        array = cls.id(dom).array
        source = list(range(len(dom), 2 * len(dom)))
        target = [i + len(right) if i < len(dom @ left)
                  else i - len(left) for i in source]
        with backend() as np:
            return cls(np.moveaxis(array, source, target), dom, cod)

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[Dim]) -> Tensor:
        xs = finset.Permutation(xs, len(doms))
        dom = cls.ob.unit().tensor(*doms)
        if xs.is_identity:
            return cls.id(dom)
        offsets = [0]
        for dim in doms:
            offsets.append(offsets[-1] + len(dim))
        axes = finset.Permutation([
            axis for i in xs for axis in range(offsets[i], offsets[i + 1])])
        source = list(range(len(dom), 2 * len(dom)))
        target = [len(dom) + axis for axis in axes.dagger()]
        cod = Dim().tensor(*(doms[i] for i in xs))
        with backend() as np:
            array = np.moveaxis(cls.id(dom).array, source, target)
        return cls(array, dom, cod)

    @classmethod
    def spider_factory(cls, n_legs_in: int, n_legs_out: int,
                       typ: Dim, phase=None) -> Tensor:
        if phase is not None:
            raise NotImplementedError
        assert_isatomic(typ, Dim)
        n, = typ.inside
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        with backend('numpy') as np:
            array = np.zeros((dom @ cod).inside, dtype=cls.dtype or int)
        for i in range(n):
            index = len(dom @ cod) * (i, )
            array[index] = array[index] + 1
        return cls(array, dom, cod)

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Dim, phase=None
                ) -> Tensor:
        """
        The tensor of interleaving spiders.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return frobenius.Diagram.spiders.__func__(  # ty: ignore
            cls, n_legs_in, n_legs_out, typ, phase)  # ty: ignore

    @classmethod
    def copy(cls, x: Dim, n: int = 2) -> Tensor:
        """
        Constructs spiders of dimension `x` with one leg in and `n` legs out.

        Parameters:
            x : The type of the spiders.
            n : The number of legs out for each spider.

        Example
        -------
        >>> from discopy import markov
        >>> n = markov.Ty('n')
        >>> F = Functor(
        ...     ob_map={n: Dim(2)}, ar_map={}, dom=markov.Diagram, dtype=int)
        >>> assert F(markov.Copy(n, 2)) == Tensor[int].copy(Dim(2), 2)\\
        ...     == Tensor[int]([1, 0, 0, 0, 0, 0, 0, 1], Dim(2), Dim(2, 2))
        """
        return cls.spiders(1, n, x)

    def transpose(self, left=False) -> Tensor:
        """
        Returns the diagrammatic transpose.

        Note
        ----
        This is *not* the same as the algebraic transpose for non-atomic dims.
        """
        # pylint: disable=unused-argument  # Dim is self-dual: one transpose
        axes = list(range(len(self.dom @ self.cod)))
        with backend() as np:
            array = np.moveaxis(self.array, axes, axes[::-1])
        return type(self)(array, self.cod[::-1], self.dom[::-1])

    l = r = property(transpose)

    def conjugate(self, diagrammatic=True) -> Tensor:
        """
        Returns the conjugate of a tensor.

        Parameters
        ----------
        diagrammatic : bool, default: True
            Whether to use the diagrammatic or algebraic conjugate.
        """
        if not diagrammatic:
            with backend() as np:
                return type(self)(np.conj(self.array), self.dom, self.cod)
        # reverse the wires for both inputs and outputs
        source = list(range(len(self.dom @ self.cod)))
        target = [
            len(self.dom) - i - 1 for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, source, target))
        return type(self)(array, self.dom[::-1], self.cod[::-1])

    @classmethod
    def zero(cls, dom: Dim, cod: Dim) -> Tensor:
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zero(Dim(2), Dim(2))\\
        ...     == Tensor([0, 0, 0, 0], Dim(2), Dim(2))
        """
        with backend() as np:
            return cls(np.zeros((dom @ cod).inside, dtype=cls.dtype or int),
                       dom, cod)

    def jacobian(self, *variables: "list[sympy.Symbol]", **params) -> Tensor:
        """
        Jacobian with respect to :code:`variables`.

        Parameters:
            variables : The list of variables to differentiate.

        Returns
        -------
        tensor : Tensor
            with :code:`tensor.dom == self.dom`
            and :code:`tensor.cod == Dim(len(variables)) @ self.cod`.

        Examples
        --------
        >>> from sympy import Expr  # doctest: +EXTRA
        >>> from sympy.abc import x, y, z
        >>> vector = Tensor[Expr]([x ** 2, y * z], Dim(1), Dim(2))
        >>> vector.jacobian(x, y, z)
        Tensor[Expr]([2*x, 0, 0, z, 0, y], dom=Dim(1), cod=Dim(3, 2))
        """
        n = len(variables) or 1
        dim = Dim(n)
        result = self.zero(self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = type(self)(
                [int(j == i) for j in range(n)], Dim(1), dim)
            result += onehot @ self.grad(var, **params)
        return result


class Functor(frobenius.Functor):
    """
    A tensor functor is a frobenius functor with a domain category ``dom``
    and ``Tensor[dtype]`` as codomain for a given ``dtype``.

    Calling it on a diagram converts it to a :class:`CMap` and contracts
    the network in a single ``einsum`` call of its ``contractor``,
    passing any optional einsum parameters through.

    Parameters:
        ob_map : The object mapping.
        ar_map : The arrow mapping.
        dom : The domain of the functor, i.e. the class of diagrams
            it evaluates, the class attribute ``dom`` by default.
        dtype : The datatype for the codomain ``Tensor[dtype]``.
        optimize : The contraction path, passed verbatim to the backend
            ``einsum``, e.g. ``"greedy"``, ``"optimal"`` or an explicit
            path.
        contractor : The backend whose ``einsum`` contracts the network,
            e.g. :class:`NumPy` or a :class:`Contractor` such as
            :class:`OptEinsum` or :class:`Quimb`. By default, the
            ``einsum`` of the active :func:`backend`, switching to
            :class:`OptEinsum` past ``config.MAX_EINSUM_INDICES`` indices.
        params : Any other optional parameter of the ``einsum`` method of
            the contractor, passed verbatim.

    Example
    -------
    >>> n, s = map(rigid.Ty, "ns")
    >>> Alice = rigid.Box('Alice', rigid.Ty(), n)
    >>> loves = rigid.Box('loves', rigid.Ty(), n.r @ s @ n.l)
    >>> Bob = rigid.Box('Bob', rigid.Ty(), n)
    >>> diagram = Alice @ loves @ Bob\\
    ...     >> rigid.Cup(n, n.r) @ s @ rigid.Cup(n.l, n)

    >>> F = Functor(
    ...     ob_map={s: 1, n: 2},
    ...     ar_map={Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]},
    ...     dom=rigid.Diagram, dtype=bool)
    >>> F(diagram)
    Tensor[bool]([True], dom=Dim(1), cod=Dim(1))

    >>> rewrite = diagram\\
    ...     .transpose_box(2).transpose_box(0, left=True).normal_form()
    >>> Equation(diagram, rewrite).draw(
    ...     figsize=(8, 3), doctest='docs/_static/tensor/rewrite.svg')

    .. image:: /_static/tensor/rewrite.svg
        :align: center

    >>> assert F(diagram) == F(rewrite)
    """
    dom, cod = frobenius.Diagram, Tensor

    def __init__(
            self, ob_map: dict[cat.Ob, Dim] | Callable[[cat.Ob], Dim],
            ar_map: dict[cat.Box, list] | Callable[[cat.Box], list],
            dom: type | None = None, dtype: type | None = float,
            optimize="greedy", contractor: Backend | None = None,
            **params):
        self.dtype, self.optimize, self.params = dtype, optimize, params
        self.contractor = contractor
        cod = type(self).cod[dtype]
        super().__init__(ob_map, ar_map, dom=dom or type(self).dom, cod=cod)

    def __repr__(self):
        dtype = None if self.dtype is None else self.dtype.__name__
        optimize = "" if self.optimize == "greedy"\
            else f", optimize={self.optimize!r}"
        contractor = "" if self.contractor is None\
            else f", contractor={self.contractor!r}"
        params = "".join(
            f", {key}={value!r}" for key, value in self.params.items())
        return factory_name(type(self))\
            + f"(ob_map={self.ob_map}, ar_map={self.ar_map}, "\
            + f"dom={factory_name(self.dom)}, "\
            + f"dtype={dtype}{optimize}{contractor}{params})"

    def to_tree(self) -> dict:
        """
        Serialise a tensor functor, its dtype in the name of its codomain,
        see :meth:`discopy.cat.Functor.to_tree`, unless it has ``params``
        or a ``contractor``.

        >>> from discopy.utils import dumps, loads
        >>> x = frobenius.Ty('x')
        >>> f = frobenius.Box('f', x, x)
        >>> F = Functor({x: 2}, {f: [[0, 1], [1, 0]]}, dtype=int)
        >>> assert loads(dumps(F)) == F and loads(dumps(F)).dtype is int
        >>> dumps(Functor({x: 2}, {}, order='C'))
        Traceback (most recent call last):
        ...
        TypeError: A tensor functor with params or a contractor has no tree: \
they go to the einsum of a backend verbatim, which a file should not reach.
        """
        if self.params or self.contractor is not None:
            raise TypeError(messages.NO_TREE_PARAMS)
        return super().to_tree() | (
            {} if self.optimize == "greedy" else {'optimize': self.optimize})

    @classmethod
    def from_tree(cls, tree: dict) -> Functor:
        """
        Decode a tensor functor, which takes its codomain's dtype, and an
        ``optimize`` of the types its backend reads, a string, a boolean or
        an explicit path.
        """
        functor = cat.Functor.from_tree(tree)
        optimize = tree.get('optimize', "greedy")
        assert_isinstance(optimize, (str, bool, list))
        return cls(functor.ob_map, functor.ar_map, dom=functor.dom,
                   dtype=functor.cod.dtype, optimize=optimize)

    def __call__(self, other):
        """
        The image of an object, a box, a diagram or a hypergraph, see
        :meth:`contract`.

        A cup or a cap on a self-dual object, as at frobenius, is the
        spider with two legs on its image, so that the functor factors
        through the hypergraph; any other is the nested
        :meth:`Tensor.cups` or :meth:`Tensor.caps`. Likewise, a rotated
        box with a self-dual object on its boundary is its box bent by
        those spiders, contracted as a map, and any other rotated box is
        the transpose of the image of its box.

        Example
        -------
        >>> n = frobenius.Ty('n')
        >>> g = frobenius.Box('g', n, n)
        >>> F = Functor({n: Dim(2, 3)}, {g: list(range(36))}, dtype=int)
        >>> cup = frobenius.Cup(n, n)
        >>> assert F(cup) == F(frobenius.Spider(2, 0, n))\\
        ...     == F(frobenius.Id(n @ n) >> cup)
        >>> assert F(g.r) == F(g.transpose()) != F(g).r

        A hypergraph of :mod:`discopy.frobenius` is contracted as it
        stands, see :meth:`spider_operands`, any other through its diagram,
        see :meth:`operands`.

        >>> graph = (g >> frobenius.Spider(1, 2, n)).to_hypergraph()
        >>> assert F(graph) == F(graph.to_diagram())
        """
        if isinstance(other, Bubble):
            return self(other.arg).map(other.func, self.dtype)
        if isinstance(other, (rigid.Cup, rigid.Cap))\
                and other.left == other.right:
            legs = (2, 0) if isinstance(other, rigid.Cup) else (0, 2)
            return self.cod.spiders(*legs, self(other.left))
        if isinstance(other, rigid.Box) and other.z and any(
                x.r == x for x in other.dom @ other.cod):
            return self.contract(cmap.CMap.from_diagram(other))
        if other is None or isinstance(other, (
                cat.Ob, cat.Box, monoidal.Colour, monoidal.Ty)):
            return super().__call__(other)
        if isinstance(other, (cmap.CMap, hypergraph.Hypergraph)):
            return self.contract(other)
        assert_isinstance(other, monoidal.Diagram)
        return self.contract(cmap.CMap.from_diagram(other))

    def operands(
            self, other: "cmap.CMap | hypergraph.Hypergraph") -> tuple[
            list, list[list[int]], list[int], list[int]]:
        """
        The Einstein notation for the image of a combinatorial map: its
        arrays, the list of integer indices of each, and the free indices
        of its domain and codomain.

        The 2-cycles of its ``edges`` involution are the summed indices,
        boxes are the tensors and the boundary ports are the free indices,
        each carried by an identity so that every index is on an array. A
        wire is one index of the size of its object's image, pairing its
        factors in order as a straight wire and a spider do, except a cup
        or a cap between two distinct adjoints, which pairs them nested,
        one index per factor, as :meth:`Tensor.cups` does. A loop is an
        identity with a repeated index, i.e. a trace.

        Parameters:
            other : The combinatorial map to translate.

        Example
        -------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> F = Functor(ob_map=lambda dim: dim,
        ...             ar_map=lambda box: box.array, dtype=int)
        >>> arrays, indices, inputs, outputs = F.operands(
        ...     (vector >> vector[::-1]).to_map())
        >>> for array, index in zip(arrays, indices):
        ...     print(f"{array.tolist()}, {index}")
        [0, 1], [0]
        [0, 1], [0]
        >>> assert inputs == outputs == []
        """
        if isinstance(other, hypergraph.Hypergraph):
            if issubclass(other.category, frobenius.Diagram):
                return self.spider_operands(other)
            other = cmap.CMap.from_diagram(other.to_diagram())
        ports, wires, fresh = other.ports, {}, count()
        for source, target in enumerate(other.edges):
            if source > target:
                continue
            dims = self(ports[source].obj).inside
            is_nested = ports[source].obj != ports[target].obj and (
                ports[source].kind.is_positive
                == ports[target].kind.is_positive)
            dims = dims if is_nested else (product(dims), )
            labels = [(next(fresh), dim) for dim in dims]
            wires[source] = labels
            wires[target] = labels[::-1] if is_nested else labels
        factors = lambda boundary: [
            factor for port in boundary for factor in wires[port]]
        arrays, indices, inputs, outputs = [], [], [], []
        with backend() as np:
            eye = lambda dim: np.array(np.eye(dim) if self.dtype is None
                                       else np.eye(dim, dtype=int),
                                       dtype=self.dtype)
            for wire, dim in factors(range(len(other.dom))):
                label = next(fresh)
                arrays.append(eye(dim))
                indices.append([label, wire])
                inputs.append(label)
            start = len(other.dom)
            for box in other.boxes:
                arity, coarity = len(box.dom), len(box.cod)
                box_ports = list(range(start, start + arity)) + list(
                    reversed(range(
                        start + arity, start + arity + coarity)))
                arrays.append(self(box).array.reshape(
                    [dim for _, dim in factors(box_ports)]))
                indices.append([label for label, _ in factors(box_ports)])
                start += arity + coarity
            for wire, dim in factors(range(
                    other.n_ports - len(other.cod), other.n_ports)):
                label = next(fresh)
                arrays.append(eye(dim))
                indices.append([wire, label])
                outputs.append(label)
            for loop in other.loops:
                arrays.append(eye(product(self(loop).inside)))
                indices.append(2 * [next(fresh)])
        return arrays, indices, inputs, outputs

    def spider_operands(self, other: "hypergraph.Hypergraph") -> tuple[
            list, list[list[int]], list[int], list[int]]:
        """
        The Einstein notation for the image of a hypergraph, as in
        :meth:`operands`: one index per spider, of the size of the image of
        its type, and one array per box, indexed by the spiders on its
        ports. A spider is one index however many ports it has, not a
        tensor. Each boundary port is a free index carried by an identity,
        and so is each port of a box after the first on the same spider, so
        that no array repeats an index, which some contractors read as a
        trace. A spider on no port is summed over alone, i.e. it counts the
        size of its type.

        Parameters:
            other : The hypergraph to translate, of :mod:`frobenius`: any
                other goes through its diagram, see :meth:`operands`.

        Example
        -------
        >>> x = frobenius.Ty('x')
        >>> f = frobenius.Box('f', frobenius.Ty(), x @ x)
        >>> F = Functor({x: 3}, {f: [[1, 1, 0], [0, 1, 1], [1, 0, 1]]},
        ...             dtype=int)
        >>> graph = frobenius.Hypergraph(
        ...     frobenius.Ty(), x, (f, ), ((), (((), (0, 0)), ), (0, )))
        >>> arrays, indices, inputs, outputs = F.spider_operands(graph)
        >>> indices
        [[1, 0], [0, 1], [0, 2]]
        >>> F(graph)
        Tensor[int]([1, 1, 1], dom=Dim(1), cod=Dim(3))
        >>> scalar = frobenius.Hypergraph(
        ...     frobenius.Ty(), frobenius.Ty(), (), ((), (), ()),
        ...     spider_types={0: x})
        >>> F(scalar)
        Tensor[int]([3], dom=Dim(1), cod=Dim(1))
        """
        dims = [product(self(typ).inside) for typ in other.spider_types]
        arrays, indices, inputs, outputs = [], [], [], []
        fresh = count(other.n_spiders)
        with backend() as np:
            def cast(unit, dim):
                return np.array(
                    unit(dim) if self.dtype is None
                    else unit(dim, dtype=int), dtype=self.dtype)
            for spider in other.dom_wires:
                inputs.append(next(fresh))
                arrays.append(cast(np.eye, dims[spider]))
                indices.append([inputs[-1], spider])
            for box, wiring in zip(other.boxes, other.box_wires):
                wires, index = sum(wiring, ()), []
                for spider in wires:
                    if spider in index:  # a diagonal, kept as one index
                        index.append(next(fresh))
                        arrays.append(cast(np.eye, dims[spider]))
                        indices.append([index[-1], spider])
                    else:
                        index.append(spider)
                arrays.append(self(box).array.reshape(
                    [dims[spider] for spider in wires]))
                indices.append(index)
            for spider in other.cod_wires:
                outputs.append(next(fresh))
                arrays.append(cast(np.eye, dims[spider]))
                indices.append([spider, outputs[-1]])
            for spider in other.scalar_spiders:
                arrays.append(cast(np.ones, dims[spider]))
                indices.append([spider])
        return arrays, indices, inputs, outputs

    def to_quimb(self, other) -> "quimb.tensor.TensorNetwork":
        """
        The quimb tensor network of the image of a diagram or a
        combinatorial map: one tensor per array of :meth:`operands`, with
        the free indices named ``inp0, ...`` and ``out0, ...`` in boundary
        order, the order in which quimb contraction outputs them.

        Parameters:
            other : The diagram or combinatorial map to translate.
        """
        if not isinstance(other, (cmap.CMap, hypergraph.Hypergraph)):
            other = cmap.CMap.from_diagram(other)
        arrays, indices, inputs, outputs = self.operands(other)
        names = {label: f"inp{i}" for i, label in enumerate(inputs)}\
            | {label: f"out{i}" for i, label in enumerate(outputs)}
        return Quimb().network(arrays, indices, names)

    def contract(
            self, other: "cmap.CMap | hypergraph.Hypergraph") -> Tensor:
        """
        Contract the image of a combinatorial map or a hypergraph, read as
        Einstein notation by :meth:`operands`, in a single ``einsum`` call
        of the ``contractor``. By default, this is the ``einsum`` of the
        active :func:`backend`, and networks with more than
        ``config.MAX_EINSUM_INDICES`` indices are contracted with
        :class:`OptEinsum` instead.

        A spider of a hypergraph is an index on any number of arrays, so
        NumPy's default bound on intermediates, the size of the largest
        operand, can forbid every pairwise step and leave one naive sum
        over all the indices: a hypergraph lifts it for the ``"greedy"``
        path, not for ``"optimal"``, whose search it would let explode.

        Parameters:
            other : The combinatorial map or the hypergraph to contract.

        Example
        -------
        >>> from discopy import compact
        >>> n = compact.Ty('n')
        >>> F = Functor({n: Dim(2, 3)}, {}, dom=compact.Diagram, dtype=int)
        >>> cup = compact.Cup(n, n.r)
        >>> assert F(cup) == F(compact.Id(n @ n.r) >> cup)\\
        ...     == Tensor[int].cups(Dim(2, 3), Dim(3, 2))
        """
        arrays, indices, inputs, outputs = self.operands(other)
        if not arrays:
            return self.cod([1], self(other.dom), self(other.cod))
        operands = [x for pair in zip(arrays, indices) for x in pair]
        n_indices = len({j for index in indices for j in index})
        contractor = self.contractor or (
            get_backend() if n_indices <= config.MAX_EINSUM_INDICES
            else OptEinsum())
        params = dict(self.params, optimize=self.optimize)\
            if isinstance(contractor, (NumPy, JAX, Contractor))\
            else self.params
        if isinstance(other, hypergraph.Hypergraph)\
                and isinstance(contractor, NumPy)\
                and self.optimize in ("greedy", True):
            params['optimize'] = ("greedy", sys.maxsize)
        array = contractor.einsum(*operands, inputs + outputs, **params)
        return self.cod(array, self(other.dom), self(other.cod))


@factory
class Diagram[dtype](TestableGeneric, frobenius.Diagram):
    """
    A tensor diagram is a frobenius diagram with tensor boxes.

    Example
    -------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> diagram = vector[::-1] >> vector @ vector
    >>> print(diagram)
    vector[::-1] >> vector >> Dim(2) @ vector

    A tensor diagram is subscripted by the ``dtype`` of the :class:`Tensor`
    it evaluates to, so numpy's scalar types are named as in
    :class:`Matrix`:

    >>> import numpy as np
    >>> assert Box[np.float64] is Box[float]
    >>> Box('f', Dim(1), Dim(1), [0.5])
    tensor.Box[float]('f', Dim(1), Dim(1), data=[0.5])
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """ None, a tensor diagram drawing no arrays. """
        return ()

    sum_factory: ClassVar[type["Sum"]]
    swap_factory: ClassVar[type["Swap"]]
    permutation_factory: ClassVar[type["Permutation"]]
    cup_factory: ClassVar[type["Cup"]]
    cap_factory: ClassVar[type["Cap"]]
    bubble_factory: ClassVar[type["Bubble"]]
    ob = Dim

    def __class_getitem__(cls, values):
        """ The dtype of a tensor, or the type parameter that declares a
        subclass, e.g. ``class Box[dtype](..., Diagram[dtype])``. """
        return super().__class_getitem__(
            values if isinstance(values, TypeVar) else Tensor[values].dtype)

    @classmethod
    def dim(cls, typ: monoidal.Ty) -> Dim:
        """
        The dimension of a type, i.e. its image under :meth:`eval`: the
        ``dim`` of each atom, or the atom itself when it is a number.

        Parameters:
            typ : A type of the category of the diagram.

        Example
        -------
        >>> from discopy.quantum import bit, qubit
        >>> assert Diagram.dim(bit @ qubit) == Dim(2, 2)
        >>> assert Diagram.dim(Dim(2, 3)) == Dim(2, 3)
        """
        return Dim(*(getattr(obj, "dim", obj) for obj in typ.inside))

    def eval(self, dtype: type | None = None, optimize="greedy",
             contractor: Backend | None = None, **params) -> Tensor:
        """
        Evaluate a tensor network as a :class:`Tensor`: call the
        :class:`Functor` that sends each type to its :meth:`dim` and each
        box to its array.

        Parameters:
            dtype : The datatype for spiders and the result,
                inferred from the boxes by default, those inside sums
                and bubbles included, i.e. the one
                :meth:`Matrix.promote` gives for their ``dtype``.
            optimize : The contraction path, passed verbatim to the
                ``einsum`` of the contractor.
            contractor : The backend whose ``einsum`` contracts the
                network, see :meth:`Functor.contract`.
            params : Any other optional parameter of the ``einsum``
                method of the contractor, passed verbatim.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).eval().array == 1
        >>> assert (vector >> vector[::-1]).eval(
        ...     optimize="optimal").array == 1
        >>> assert (vector >> vector[::-1]).eval(  # doctest: +EXTRA
        ...     contractor=Quimb()).array == 1
        >>> (vector >> vector[::-1]).eval()
        Tensor[int]([1], dom=Dim(1), cod=Dim(1))
        >>> (vector @ Box('half', Dim(1), Dim(1), [0.5])).eval()
        Tensor[float]([0. , 0.5], dom=Dim(1), cod=Dim(2))
        """
        category = self.category if isinstance(self, cmap.CMap) else self.ar
        dtype = dtype or getattr(self, "dtype", None)\
            or getattr(category, "dtype", None)
        boxes = list(self.boxes) if dtype is None else []
        for box in boxes:
            inside = box.terms if isinstance(box, Sum)\
                else (box.arg, ) if isinstance(box, Bubble) else ()
            boxes += [b for diagram in inside for b in diagram.boxes]
        typed = [Matrix[box.dtype].zero(0, 0)
                 for box in boxes if box.dtype is not None]
        if typed:
            dtype = typed[0].promote(*typed[1:]).dtype
        return Functor(
            ob_map=category.dim, ar_map=lambda box: box.array,
            dtype=dtype, optimize=optimize, contractor=contractor,
            **params)(self)

    def to_quimb(
            self, dtype: type | None = None) -> "quimb.tensor.TensorNetwork":
        """
        Convert a tensor diagram to a quimb tensor network, see
        :meth:`Functor.to_quimb`: swaps, cups and caps are wiring and the
        free indices are named ``inp0, ...`` and ``out0, ...`` in
        boundary order.

        Parameters:
            dtype : The datatype for spiders and boundary identities,
                inferred from the boxes by default.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> t_net = (vector >> vector[::-1]).to_quimb()  # doctest: +EXTRA
        >>> assert t_net.contract(preserve_tensor=True).data == 1
        >>> t_net = vector[::-1].to_quimb(dtype=complex)  # doctest: +EXTRA
        >>> assert t_net.contract(preserve_tensor=True).data.dtype == complex
        """
        if isinstance(self, Sum):
            return Box("sum", self.dom, self.cod, self.eval(dtype).array)\
                .to_quimb(dtype)
        category = self.category if isinstance(self, cmap.CMap) else self.ar
        return Functor(
            ob_map=category.dim, ar_map=lambda box: box.array,
            dtype=dtype or getattr(self, "dtype", None)).to_quimb(self)

    def to_tn(self, dtype: type | None = None) -> tuple[
            list["tensornetwork.AbstractNode"], list["tensornetwork.Edge"]]:
        """
        Convert a tensor diagram to :code:`tensornetwork`.

        Parameters:
            dtype : Used for spiders.

        Examples
        --------
        >>> import numpy as np
        >>> from tensornetwork import Node, Edge  # doctest: +EXTRA
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> nodes, output_edge_order = vector.to_tn()
        >>> node, = nodes
        >>> assert node.name == "vector" and np.all(node.tensor == [0, 1])
        >>> assert output_edge_order == [node[0]]
        """
        import tensornetwork as tn
        if dtype is None:
            dtype = self.dtype
        nodes = [
            tn.CopyNode(
                2, dim, f'input_{i}',
                dtype=dtype)  # ty: ignore[invalid-argument-type]
            for i, dim in enumerate(self.dim(self.dom).inside)]
        inputs, outputs = [n[0] for n in nodes], [n[1] for n in nodes]
        for box, offset in zip(self.boxes, self.offsets):
            if isinstance(box, Permutation):
                segment = outputs[offset:offset + len(box.dom)]
                outputs[offset:offset + len(box.dom)] = [
                    segment[i] for i in box.perm]
                continue
            dims = (len(box.dom), len(box.cod))
            if isinstance(box, Cup) or isinstance(box, Spider)\
                    and box.phase is None and dims != (0, 0):
                if dims == (1, 1):  # identity
                    continue
                elif dims == (2, 0):  # cup
                    tn.connect(outputs[offset], outputs[offset + 1])
                    del outputs[offset:offset + 2]
                    continue
                else:
                    node = tn.CopyNode(
                        sum(dims), box.typ.inside[0],
                        dtype=dtype)  # ty: ignore[invalid-argument-type]
            else:
                array = box.eval(dtype=dtype).array
                node = tn.Node(array, str(box))
            for i, _ in enumerate(box.dom):
                tn.connect(outputs[offset + i], node[i])
            outputs[offset:offset + len(box.dom)] = node[len(box.dom):]
            nodes.append(node)  # ty: ignore[invalid-argument-type]
        return nodes, inputs + outputs  # ty: ignore[invalid-return-type]

    def grad(self, var, **params):
        """ Gradient with respect to :code:`var`. """
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        if len(self.inside[0].boxes) > 1:
            return (self[:1].to_staircases() >> self[1:]).grad(var, **params)
        box, offset, dom = self.boxes[0], self.offsets[0], self.dom
        left, right = dom[:offset], dom[offset + len(box.dom):]
        tail = self[1:]
        t1 = self.id(left) @ box.grad(var, **params) @ self.id(right) >> tail
        t2 = self.id(left) @ box @ self.id(right) >> tail.grad(var, **params)
        return t1 + t2

    def jacobian(self, variables, **params) -> Diagram:
        """
        Diagrammatic jacobian with respect to :code:`variables`.

        Parameters
        ----------
        variables : List[sympy.Symbol]
            Differentiated variables.

        Returns
        -------
        diagram : Sum
            with :code:`diagram.dom == self.dom`
            and :code:`diagram.cod == Dim(len(variables)) @ self.cod`,
            whose :meth:`eval` is the :meth:`Tensor.jacobian` of
            :code:`self.eval()`.

        Examples
        --------
        >>> from sympy import Expr  # doctest: +EXTRA
        >>> from sympy.abc import x, y, z
        >>> vector = Box("v", Dim(1), Dim(2), [x ** 2, y * z])
        >>> vector.jacobian([x, y, z]).eval(dtype=Expr)
        Tensor[Expr]([2*x, 0, 0, z, 0, y], dom=Dim(1), cod=Dim(3, 2))
        """
        n = len(variables) or 1
        dim = Dim(n)
        result = Sum((), self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = [int(j == i) for j in range(n)]
            result += Box(str(var), Dim(1), dim, onehot)\
                @ self.grad(var, **params)
        return result

    # A tensor box carries an array, which the strategy of a free category
    # does not draw: tensor diagrams, circuits and ZX diagrams stay out of
    # the matrix.
    strategy = no_strategy


CMap = cmap.CMap[Diagram]


class Box[dtype](frobenius.Box, Diagram[dtype]):
    """
    A tensor box is a frobenius box with an array as data.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. its input dimension.
        cod : The codomain of the box, i.e. its output dimension.
        data : The array inside the tensor box.

    Example
    -------
    >>> b1 = Box('sauce_0', Dim(1), Dim(2), data=[0.84193562, 0.91343221])
    >>> b1.eval()
    Tensor[float]([0.84193562, 0.91343221], dom=Dim(1), cod=Dim(2))
    """

    def __setstate__(self, state):
        if "data" not in state and state.get("_array", None) is not None:
            state['data'] = state['_array']
            del state["_array"]
        super().__setstate__(state)
        if self.dtype is None and self.data is not None:
            self.data, dtype = self._get_data_dtype(self.data)
            self.__class__ = self.__class__[dtype]

    def __new__(
            cls, name=None, dom=None, cod=None, data=None, *args, **kwargs):
        if cls.dtype is not None or data is None:
            return object.__new__(cls)
        data, dtype = cls._get_data_dtype(data)
        return cls.__new__(
            cls[dtype], name, dom, cod, data, *args, **kwargs)

    @staticmethod
    def _get_data_dtype(data):
        with backend() as np:
            data = np.array(data)
            # The dtype of an np.arrays is a class that contains a type
            # attribute that is the actual type. However, other backends
            # have different structures, so this is the easiest option:
            dtype = getattr(data.dtype, "type", data.dtype)
            return data, dtype

    @property
    def array(self):
        if self.data is not None:
            with backend() as np:
                return np.array(self.data).reshape(
                    self.dim(self.dom @ self.cod).inside)

    def grad(self, var, **params):
        return self.bubble(
            func=Diff(var), drawing_name=f"$\\partial {var}$")

    def setoid(self):
        """
        Compare boxes by turning their internal ``data`` into the tuple of its
        entries: a list as it is, an array of any backend through its
        ``tolist`` and a list of lists flattened.

        Every nan entry becomes the one :data:`math.nan`, of any number type
        and inside tuples too, and a complex one with a non-zero imaginary
        part the pair of its parts, so that a box holding nan equals itself
        and its hash is stable: the data labels a generator, it is not a
        number being compared. The winding number ``z`` is compared too, as
        in :meth:`discopy.rigid.Box.setoid`, and the time step, as in
        :meth:`discopy.feedback.Box.setoid`. The ``dtype`` is compared by its
        :func:`discopy.utils.factory_name`, a class object hashing as its
        address and a backend's descriptor, e.g. ``torch.float64``, being no
        class at all.

        >>> import numpy as np
        >>> f = Box('f', Dim(2), Dim(2), np.array([[0, 1], [1, 0]]))
        >>> assert f == Box('f', Dim(2), Dim(2), [[0, 1], [1, 0]])\\
        ...     == Box('f', Dim(2), Dim(2), [0, 1, 1, 0])
        >>> assert f != Box('f', Dim(2), Dim(2), [0, 1, 1, 1])
        >>> assert len({f, f.dagger().dagger()}) == 1
        >>> g = Box('g', Dim(2), Dim(2), np.array([[np.nan, 1], [0, 1]]))
        >>> assert g == g and len({g, g.dagger().dagger()}) == 1
        """
        def canonical(x):
            if isinstance(x, (list, tuple)):
                return tuple(map(canonical, x))
            # pylint: disable-next=comparison-with-itself  # false only for nan
            if not isinstance(x, numbers.Complex) or x == x:
                return x
            if isinstance(x, numbers.Real) or x.imag == 0:
                return math.nan
            return canonical(x.real), canonical(x.imag)

        data = getattr(self.data, "tolist", lambda: self.data)()
        if isinstance(data, list) and any(isinstance(x, list) for x in data):
            with backend('numpy') as np:
                data = np.array(data).reshape(-1).tolist()
        entries = () if data is None\
            else tuple(data) if isinstance(data, list) else (data, )
        dtype = None if self.dtype is None else factory_name(self.dtype)
        return (self.name, self.dom, self.cod, dtype, self.is_dagger,
                self.z, self.deterministic, self.affine, self.time_step)\
            + tuple(map(canonical, entries))

    def __repr__(self):
        """
        The representation of a tensor box, with its ``data`` as the nested
        list of its entries through :func:`repr_nonfinite`, which the box
        equals by :meth:`setoid` whatever array it holds.

        >>> import numpy as np
        >>> f = Box('f', Dim(1), Dim(2), np.array([np.nan, -np.inf]))
        >>> print(repr(f).split('data=')[1])
        [float('nan'), float('-inf')])
        >>> from discopy import tensor
        >>> assert eval(repr(f), {'tensor': tensor, 'Dim': Dim}) == f
        """
        if self.data is None:
            return super().__repr__()
        data = getattr(self.data, "tolist", lambda: self.data)()
        return super().__repr__().replace(
            f", data={self.data!r}", f", data={repr_nonfinite(data)}", 1)


class Cup(frobenius.Cup, Box):
    """
    A tensor cup is a frobenius cup in a tensor diagram.

    Parameters:
        left (Dim) : The atomic type.
        right (Dim) : Its adjoint.
    """


class Cap(frobenius.Cap, Box):
    """
    A tensor cap is a frobenius cap in a tensor diagram.

    Parameters:
        left (Dim) : The atomic type.
        right (Dim) : Its adjoint.
    """


class Permutation[dtype](frobenius.Permutation, Box[dtype]):
    "A permutation in a tensor diagram."

    @property
    def array(self):
        return Tensor.permutation(
            self.perm, [self.dim(x) for x in self.dom]).array


class Swap[dtype](Permutation[dtype], frobenius.Swap, Box[dtype]):
    """
    A tensor swap is a frobenius swap in a tensor diagram.

    Parameters:
        left (Dim) : The type on the top left and bottom right.
        right (Dim) : The type on the top right and bottom left.
    """


class Spider[dtype](frobenius.Spider, Box[dtype]):
    """
    A tensor spider is a frobenius spider in a tensor diagram.

    Parameters:
        n_legs_in (int) : The number of legs in.
        n_legs_out (int) : The number of legs out.
        typ (Dim) : The dimension of the spider.
        data : The phase of the spider.

    Examples
    --------
    >>> vector = Box('vec', Dim(1), Dim(2), [0, 1])
    >>> spider = Spider(1, 2, Dim(2))
    >>> assert (vector >> spider).eval() == (vector @ vector).eval()
    >>> Equation(vector >> spider, vector @ vector).draw(figsize=(3, 2),
    ...     doctest='docs/_static/tensor/frobenius-example.svg')

    .. image:: /_static/tensor/frobenius-example.svg
        :align: center
    """
    @property
    def array(self):
        """
        The array of a spider without a phase, see
        :meth:`Tensor.spider_factory`; a phase has no array in
        :class:`Tensor` yet, so a phased spider raises rather than reading
        its phase as its array.
        """
        if self.phase is not None:
            raise NotImplementedError(
                f"{self} has a phase, which Tensor does not interpret.")
        return Tensor.spiders(
            len(self.dom), len(self.cod),
            self.typ).array  # ty: ignore[invalid-argument-type]


class Sum[dtype](frobenius.Sum, Box[dtype]):
    """
    A formal sum of tensor diagrams with the same domain and codomain, a
    frobenius sum with the flags of a :class:`discopy.markov.Sum`.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Dim) : The domain of the formal sum.
        cod (Dim) : The codomain of the formal sum.

    Example
    -------
    >>> f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
    >>> assert f.is_cartesian and not (f + f).is_deterministic
    """


def func_repr(func: Callable) -> str:
    """
    The repr of a function in a tensor bubble or a derivative: a named
    function prints as its import path, so that it reads back with
    :func:`eval`, anything else, e.g. a lambda, as its own repr.

    Example
    -------
    >>> import numpy
    >>> assert func_repr(abs) == "abs"
    >>> assert func_repr(numpy.sqrt) == "numpy.sqrt"
    >>> assert func_repr(lambda x: x).startswith("<function")
    """
    name = getattr(func, "__qualname__", "<")
    return repr(func) if "<" in name\
        else f"{func.__module__}.{name}".removeprefix("builtins.")


class Bubble(feedback.Bubble, Box):
    """
    Bubble in a tensor diagram, applies a function elementwise. It is a
    feedback bubble, so that a circuit bubble delays as one.

    Parameters
    ----------
    inside : tensor.Diagram
        The diagram inside the bubble.
    func : callable
        The function to apply, default is :code:`lambda x: int(not x)`.
        A named function prints as its import path, e.g. :code:`abs` or
        :code:`numpy.sqrt`, so the bubble reads back from its repr, a
        lambda does not.

    Examples
    --------

    >>> men = Box("men", Dim(1), Dim(2), [0, 1])
    >>> mortal = Box("mortal", Dim(2), Dim(1), [1, 1])
    >>> men_are_mortal = (men >> mortal.bubble()).bubble()
    >>> assert men_are_mortal.eval(dtype=bool)
    >>> men_are_mortal.draw(wire_labels=False,
    ...                     doctest='docs/_static/tensor/men-are-mortal.svg')

    .. image:: /_static/tensor/men-are-mortal.svg
        :align: center

    >>> from sympy import Expr  # doctest: +EXTRA
    >>> from sympy.abc import x
    >>> f = Box('f', Dim(2), Dim(2), [1, 0, 0, x])
    >>> g = Box('g', Dim(2), Dim(2), [-x, 0, 0, 1])
    >>> def grad(diagram, var):
    ...     return diagram.bubble(
    ...         func=lambda x: getattr(x, "diff", lambda _: 0)(var),
    ...         drawing_name=f"d${var}$" )
    >>> lhs = grad(f >> g, x)
    >>> rhs = (grad(f, x) >> g) + (f >> grad(g, x))
    >>> assert lhs.eval(dtype=Expr) == rhs.eval(dtype=Expr)

    >>> Equation(lhs, rhs).draw(figsize=(5, 2), wire_labels=False,
    ...                         doctest='docs/_static/tensor/product-rule.svg')

    .. image:: /_static/tensor/product-rule.svg
        :align: center
    """
    serialised_attrs = feedback.Bubble.serialised_attrs + ("func", )
    func = staticmethod(lambda x: int(not x))

    def __new__(cls, *args, data=None, **kwargs):
        return super().__new__(cls, data=data)

    def __init__(self, inside, func=None, **params):
        if func is not None:
            self.func = func
        super().__init__(inside, **params)

    def setoid(self):
        """
        Tensor bubbles are equal when their args, methods and funcs are, and
        they are equal as tensor boxes: their daggers too, and their data by
        its entries, so that a bubble holding a list hashes as a box does.

        >>> import numpy as np
        >>> f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
        >>> assert f.bubble(data=np.array([1, 2])) == f.bubble(data=[1, 2])
        >>> assert len({f.bubble(data=[1, 2]), f.bubble(data=[1, 3])}) == 2
        """
        args = tuple(f.setoid() for f in self.args)
        return (factory_key(type(self)), self.method, self.func)\
            + args + Box.setoid(self)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        if self.func is type(self).func:
            return super().__repr__()
        return super().__repr__()[:-1] + f", func={func_repr(self.func)})"

    def grad(self, var, **params):
        """
        The gradient of a bubble is given by the chain rule.

        >>> from sympy.abc import x  # doctest: +EXTRA
        >>> g = Box('g', Dim(2), Dim(2), [2 * x, 0, 0, x + 1])
        >>> f = lambda d: d.bubble(func=lambda x: x ** 2, drawing_name="f")
        >>> lhs, rhs = Box.grad(f(g), x), f(g).grad(x)

        >>> Equation(lhs, rhs).draw(wire_labels=False,
        ...     doctest='docs/_static/tensor/chain-rule.svg')

        .. image:: /_static/tensor/chain-rule.svg
            :align: center
        """
        from sympy import Symbol
        name = "$\\frac{{\\partial {}}}{{\\partial {}}}$"
        return Diagram.spiders(1, 2, self.dom)\
            >> self.arg.bubble(
                func=Diff(Symbol("tmp"), self.func),
                drawing_name=name.format(self.drawing_name, var))\
            @ self.arg.grad(var) >> Diagram.spiders(2, 1, self.cod)


class Diff:
    """
    The derivative with respect to a variable, applied to each entry of a
    tensor by the bubble of :meth:`Box.grad`, an entry with no ``diff``, e.g.
    a number, being a constant. Given a ``func``, it is the derivative of that
    function instead, bound by the variable, as the chain rule of
    :meth:`Bubble.grad` applies it.

    Parameters:
        var : The variable.
        func : The function to differentiate, if any.

    Example
    -------
    >>> from sympy.abc import x, y  # doctest: +EXTRA
    >>> assert Diff(x)(x * y) == y and Diff(x)(42) == 0
    >>> assert Diff(y, lambda z: z ** 3)(x) == 3 * x ** 2

    Diffs are equal when their variables and functions are, so that gradients
    are too, and they read back from their repr when their function is named,
    see :func:`func_repr`.

    >>> from discopy import tensor
    >>> assert Diff(x) == eval(repr(Diff(x))) != Diff(y)
    >>> assert eval(repr(Diff(x, abs))) == Diff(x, abs)
    >>> f = Box('f', Dim(2), Dim(2), [x, 0, 0, y])
    >>> assert f.grad(x) == f.grad(x) != f.grad(y)
    """
    def __init__(self, var, func=None):
        self.var, self.func = var, func

    def __call__(self, entry):
        if self.func is None:
            return getattr(entry, "diff", lambda _: 0)(self.var)
        return self.func(self.var).diff(self.var).subs(self.var, entry)

    def __eq__(self, other):
        return isinstance(other, Diff)\
            and (self.var, self.func) == (other.var, other.func)

    def __hash__(self):
        return hash((self.var, self.func))

    def __repr__(self):
        func = "" if self.func is None else f", {func_repr(self.func)}"
        return f"{factory_name(type(self))}({self.var!r}{func})"


class Term:
    """
    A lambda term whose types are dimensions, i.e. a term in the internal
    language of the compact closed category of tensors, where the exponential
    ``x >> y`` of dimensions is ``x.r @ y`` and ``y << x`` is ``y @ x.l``.

    A dimension called on a function is an :class:`Abstraction` and on a
    string a :class:`Constant`, a term called on a term is an
    :class:`Application` and a tuple of terms is a :class:`Tuple`:

    >>> N, S = Dim(2), Dim(3)
    >>> sleeps = Constant("sleeps", S << N, data=[[1, 2], [3, 4], [5, 6]])
    >>> John = Constant("John", N, data=[1, 0])
    >>> sleeps_ = N(lambda x: sleeps(x))
    >>> assert sleeps_.cod == sleeps.cod == S @ N
    >>> assert sleeps_(John).eval().eval() == sleeps(John).eval().eval()

    Terms need not be linear: a variable occurring ``n`` times is copied with
    :meth:`Diagram.copy` and one that does not occur is discarded with
    :meth:`Diagram.discard`, those of the diagram class the term is evaluated
    in, so that a subclass chooses its own copy for every variable at once.

    >>> twice = N(lambda x: (sleeps(x), sleeps(x)))(John)
    >>> twice.draw(figsize=(4, 4), doctest="docs/_static/tensor/twice.svg")

    .. image:: /_static/tensor/twice.svg
        :align: center

    Attributes:
        cod (Dim) : The type of the term.
    """
    cod: Dim

    def __call__(self, other: Term, left: bool = False) -> Application:
        return Application(other, self, left) if left\
            else Application(self, other)

    @property
    def freevars(self) -> list[Variable]:
        """ The free variables of the term, in order of first occurrence. """
        raise NotImplementedError

    def occurrences(self, var: Variable) -> int:
        """ The number of free occurrences of a variable in the term. """
        raise NotImplementedError

    def build(self, cod: type[Diagram], env: dict) -> tuple:
        """
        The output wires of the term inside :meth:`Diagram.from_callable`,
        given an iterator over the copies of the wires of each free variable.
        """
        raise NotImplementedError

    def eval(self, cod: type[Diagram] | None = None) -> monoidal.Diagram:
        """
        The diagram of a term, from its free variables to its type.

        Parameters:
            cod : The diagram class, :class:`Diagram` by default, whose
                :meth:`Diagram.copy` and :meth:`Diagram.discard` share the
                variables.
        """
        cod = cod or Diagram
        variables = self.freevars

        def func(*nodes):
            env, i = {}, 0
            for var in variables:
                n = len(var.cod)
                env[var] = Term.share(
                    cod, var, nodes[i:i + n], self.occurrences(var))
                i += n
            return self.build(cod, env)

        dom = Dim(1).tensor(*(var.cod for var in variables))
        return cod.from_callable(dom, self.cod)(func)

    @staticmethod
    def share(cod: type[Diagram], var: Variable, nodes: tuple, uses: int):
        """ The copies of the wires of a variable, one for each use. """
        if not uses:
            cod.discard(var.cod)(*nodes)  # ty: ignore[call-non-callable]
            return iter(())
        if uses == 1:
            return iter([nodes])
        n = len(var.cod)
        copy = cod.copy(var.cod, uses)
        copies = Term.wires(
            copy(*nodes), uses * n)  # ty: ignore[call-non-callable]
        return iter([copies[i * n:(i + 1) * n] for i in range(uses)])

    @staticmethod
    def wires(outputs, n: int) -> tuple:
        """ The ``n`` output wires of a diagram called on wires. """
        return (outputs, ) if n == 1 else tuple(outputs)

    def draw(self, **params):
        """ Drawing a term is drawing its diagram, see :meth:`eval`. """
        return self.eval().draw(**params)

    def __eq__(self, other):
        return type(self) is type(other) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    @property
    def key(self) -> tuple:
        """ What makes two terms of the same class equal. """
        raise NotImplementedError


class Constant(Term):
    """
    A constant term of a given type, evaluated to a state with its data.

    Parameters:
        name : The name of the constant.
        cod : Its type.
        data : Its array.
    """
    def __init__(self, name: str, cod: Dim, data=None):
        self.name, self.cod, self.data = name, cod, data

    freevars = property(lambda self: [])

    def occurrences(self, var):
        return 0

    def build(self, cod, env):
        box = cod.box_factory(self.name, Dim(1), self.cod, self.data)
        return Term.wires(box(), len(self.cod))

    key = property(lambda self: (self.name, self.cod))

    def __repr__(self):
        data = "" if self.data is None else f", data={self.data!r}"
        return f"{factory_name(type(self))}({self.name!r}, {self.cod!r}{data})"

    def __str__(self):
        return self.name


class Variable(Term):
    """
    A variable with a name and a type.

    Parameters:
        name : The name of the variable.
        cod : Its type.
    """
    def __init__(self, name: str, cod: Dim):
        self.name, self.cod = name, cod

    freevars = property(lambda self: [self])

    def occurrences(self, var):
        return int(self == var)

    def build(self, cod, env):
        return next(env[self])

    key = property(lambda self: (self.name, self.cod))

    def __repr__(self):
        return f"{factory_name(type(self))}({self.name!r}, {self.cod!r})"

    __str__ = Constant.__str__


class Application(Term):
    """
    The application ``func(args)`` of a term of type ``y << x``, i.e.
    ``y @ x.l``, to a term of type ``x``, its last legs cupped with the
    argument, or ``args(func, left=True)`` of one of type ``x >> y``, i.e.
    ``x.r @ y``, its first legs cupped.

    Parameters:
        func : The function.
        args : The argument.
        left : Whether the argument comes in from the left.
    """
    def __init__(self, func: Term, args: Term, left: bool = False):
        assert_isinstance(func, Term)
        assert_isinstance(args, Term)
        n = len(args.cod)
        exponent = func.cod[:n] if left else func.cod[len(func.cod) - n:]
        if exponent != (args.cod.r if left else args.cod.l):
            raise TypeError(
                f"Expected {func} to take {args.cod}, got {func.cod}.")
        self.func, self.args, self.left = func, args, left
        self.cod = func.cod[n:] if left else func.cod[:len(func.cod) - n]

    freevars = property(lambda self: list(dict.fromkeys(
        self.func.freevars + self.args.freevars)))

    def occurrences(self, var):
        return self.func.occurrences(var) + self.args.occurrences(var)

    def build(self, cod, env):
        func, args = self.func.build(cod, env), self.args.build(cod, env)
        n, x = len(args), self.args.cod
        if self.left:
            cod.cups(x, x.r)(*args, *func[:n])
            return func[n:]
        cod.cups(x.l, x)(*func[len(func) - n:], *args)
        return func[:len(func) - n]

    key = property(lambda self: (self.func, self.args, self.left))

    def __repr__(self):
        left = ", left=True" if self.left else ""
        return f"{factory_name(type(self))}" \
            f"({self.func!r}, {self.args!r}{left})"

    def __str__(self):
        return f"{self.args}({self.func}, left=True)" if self.left\
            else f"{self.func}({self.args})"


class Abstraction(Term):
    """
    The abstraction of a variable of type ``x`` in a term of type ``y``,
    of type ``y << x`` with caps on its last legs, or ``x >> y`` with caps on
    its first legs if ``left``. A body given as a tuple is a :class:`Tuple`.

    Parameters:
        var : The variable.
        body : The body.
        left : Whether the variable comes in from the left.
    """
    def __init__(self, var: Variable, body: Term | tuple, left=False):
        body = Tuple(*body) if isinstance(body, tuple) else body
        self.var, self.body, self.left = var, body, left
        self.cod = var.cod >> body.cod if left else body.cod << var.cod

    freevars = property(lambda self: [
        var for var in self.body.freevars if var != self.var])

    def occurrences(self, var):
        return 0 if var == self.var else self.body.occurrences(var)

    def build(self, cod, env):
        x = self.var.cod
        caps = cod.caps(x.r, x) if self.left else cod.caps(x, x.l)
        caps = Term.wires(caps(), 2 * len(x))
        wires, legs = (caps[len(x):], caps[:len(x)]) if self.left\
            else (caps[:len(x)], caps[len(x):])
        body = self.body.build(cod, env | {self.var: Term.share(
            cod, self.var, wires, self.body.occurrences(self.var))})
        return legs + body if self.left else body + legs

    key = property(lambda self: (self.var, self.body, self.left))

    def __repr__(self):
        left = ", left=True" if self.left else ""
        return f"{factory_name(type(self))}" \
            f"({self.var!r}, {self.body!r}{left})"

    def __str__(self):
        left = ", left=True" if self.left else ""
        return f"{self.var.cod!r}(lambda {self.var}{left}: {self.body})"


class Tuple(Term):
    """
    A tuple of terms, of the tensor of their types.

    Parameters:
        terms : The terms.
    """
    def __init__(self, *terms: Term):
        self.terms, self.cod = terms, Dim(1).tensor(*(t.cod for t in terms))

    freevars = property(lambda self: list(dict.fromkeys(
        var for term in self.terms for var in term.freevars)))

    def occurrences(self, var):
        return sum(term.occurrences(var) for term in self.terms)

    def build(self, cod, env):
        return sum((tuple(term.build(cod, env)) for term in self.terms), ())

    key = property(lambda self: self.terms)

    def __repr__(self):
        terms = ', '.join(map(repr, self.terms))
        return f"{factory_name(type(self))}({terms})"

    def __str__(self):
        return f"({', '.join(map(str, self.terms))})"


Dim.variable_factory = Variable  # ty: ignore[invalid-assignment]
Dim.constant_factory = Constant  # ty: ignore[invalid-assignment]
Dim.application_factory = Application  # ty: ignore[invalid-assignment]
Dim.abstraction_factory = Abstraction  # ty: ignore[invalid-assignment]

Diagram.box_factory = Box
Diagram.sum_factory, Diagram.swap_factory = Sum, Swap
Diagram.permutation_factory = Permutation
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.spider_factory = Spider  # ty: ignore[invalid-assignment]
Diagram.bubble_factory = Bubble
Id = Diagram.id


class Equation(frobenius.Equation):
    """ The :class:`frobenius.Equation` of tensor diagrams. """
