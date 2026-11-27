"""
The category of matrices with the direct sum as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Backend
    NumPy
    JAX
    PyTorch
    TensorFlow

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        backend
        set_backend
        is_field
        is_complex
        get_backend

See also
--------

* :class:`~discopy.tensor.Tensor` is a subclass of :class:`Matrix` with the
  Kronecker product as tensor.

"""
from __future__ import annotations

import cmath
import re
from contextlib import contextmanager
from math import isfinite, isqrt, prod
from operator import index
from types import ModuleType
from collections.abc import Sequence
from typing import Literal, Callable, TYPE_CHECKING

from discopy import config, messages
from discopy.abc import (
    ComarkovCategory, DaggerMonoidalCategory, MarkovCategory, Nat)
from discopy.testable import Serialisable, TestableGeneric
from discopy.cat import (
    factory,
    assert_iscomposable,
)
from discopy.utils import (
    assert_isinstance, assert_isparallel, factory_key, factory_name,
    from_tree, unbiased)

if TYPE_CHECKING:
    import sympy
    import quimb.tensor
    from discopy.kleisli.monad import Monad
    from discopy.monad.algebra import Algebra


@factory
class Matrix[dtype](
        MarkovCategory, ComarkovCategory, DaggerMonoidalCategory,
        TestableGeneric, Serialisable):
    """
    A matrix is an ``array`` with natural numbers as ``dom`` and ``cod``.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            promote
            zero
            swap
            transpose
            conjugate
            dagger
            map
            round
            copy
            discard
            merge
            noise
            basis
            repeat
            resolvent
            trace
            lambdify
            subs
            grad

    Note
    ----
    The class ``Matrix[dtype]`` has arrays with entries in any given ``dtype``.
    For example:

    >>> Matrix[complex].id(1)
    Matrix[complex]([1.+0.j], dom=Nat(1), cod=Nat(1))
    >>> assert Matrix[complex].id(1) != Matrix[float].id(1)

    The default data type is determined by underlying array datastructure of
    the backend used. An array is initialised with ``array`` as parameter and
    the dtype of the ``Matrix`` object is the data type of this array. A
    ``dtype`` names the ring of entries, so the scalar type numpy gives each
    of ``bool``, ``int``, ``float`` and ``complex`` is that Python type and
    the class a matrix infers is the one a user names.

    >>> import numpy as np
    >>> assert Matrix[np.int64] is Matrix[int]
    >>> assert Matrix([1, 0], dom=1, cod=2).dtype == int
    >>> assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == float
    >>> assert Matrix([0.5j], dom=1, cod=1).dtype == complex
    >>> assert Matrix[int].id(2) == Matrix.id(2)

    Other precisions keep numpy's names.

    >>> Matrix([1, 2], 1, 2).cast(np.float32)
    Matrix[float32]([1., 2.], dom=Nat(1), cod=Nat(2))

    Composition promotes the data type as the backend does, see
    :meth:`promote`, so that the integer identity is a unit for a float
    matrix rather than casting it to integers.

    >>> m = Matrix([[.5, .5], [.25, .75]], 2, 2)
    >>> assert Matrix.id(2) >> m == m == m >> Matrix.id(2)

    The data type needs to have the structure of a rig (riNg with no negatives)
    i.e. with methods ``__add__`` and ``__mul__`` as well as an ``__init__``
    that can accept both ``0`` and ``1`` as input.

    Examples
    --------
    >>> m = Matrix([0, 1, 1, 0], 2, 2)
    >>> v = Matrix([0, 1], 1, 2)
    >>> v >> m >> v.dagger()
    Matrix[int]([0], dom=Nat(1), cod=Nat(1))
    >>> m + m
    Matrix[int]([0, 2, 2, 0], dom=Nat(2), cod=Nat(2))
    >>> assert m.then(m, m, m, m) == m >> m >> m >> m >> m

    The monoidal product for :py:class:`.Matrix` is the direct sum:

    >>> x = Matrix([2, 4], 2, 1)
    >>> x.array
    array([[2],
           [4]])
    >>> x @ x
    Matrix[int]([2, 0, 4, 0, 0, 2, 0, 4], dom=Nat(4), cod=Nat(2))
    >>> (x @ x).array
    array([[2, 0],
           [4, 0],
           [0, 2],
           [0, 4]])

    A matrix reads back from its representation up to the rounding of its
    entries and only up to ``config.NUMPY_THRESHOLD`` of them, past which
    NumPy elides the array, so that :meth:`repr_transparency` is weakened to
    those:

    >>> Matrix[int]([0] * 24, 4, 6)
    Matrix[int]([0, 0, 0, ..., 0, 0, 0], dom=Nat(4), cod=Nat(6))

    A matrix is immutable, and its array is not copied when it already has
    the right ``dtype``: a NumPy array is shared with the caller through a
    view that is read-only, so that writing into ``m.array`` raises while
    the caller's own array stays writeable.

    >>> a = np.zeros((2, 2), dtype=int)
    >>> m = Matrix[int](a, 2, 2)
    >>> assert np.shares_memory(m.array, a)
    >>> assert not m.array.flags.writeable and a.flags.writeable

    JAX and TensorFlow arrays are immutable anyway, but a PyTorch tensor has
    no such flag: the matrix shares it with the caller, who must not write
    into it after handing it over.
    """
    @classmethod
    def checked_params(cls) -> tuple:
        """ Integer matrices, whose entries every ``dtype`` reads exactly. """
        return (int, )

    ob = Nat

    repr_transparency = Serialisable.repr_transparency.weaken(
        max_entries=config.NUMPY_THRESHOLD)

    @classmethod
    def strategy(cls, *, dom=None, cod=None, max_size=3, max_entry=3,
                 max_entries=None):
        """
        Generate matrices with entries in ``range(max_entry + 1)`` between
        the given dimensions, or between dimensions up to ``max_size``, and
        with at most ``max_entries`` entries if given: small integers are
        exact in every ``dtype``, so the laws are checked without rounding.
        The bare class generates ``Matrix[int]``, the class it infers from
        integer entries, even for an empty matrix, whose array numpy reads
        as floats.
        """
        from hypothesis import strategies as st

        typed = cls[cls.dtype or int]
        dims = cls.ob.strategy(max_size=max_size)

        @st.composite
        def matrices(draw):
            """ Draw the shape, then one entry per coefficient. """
            shape = tuple(draw(dims) if x is None else x for x in (dom, cod))
            size = index(shape[0]) * index(shape[1])
            entries = st.lists(
                st.integers(min_value=0, max_value=max_entry),
                min_size=size, max_size=size)
            return typed(draw(entries), *shape)

        return matrices() if max_entries is None else matrices().filter(
            lambda matrix: index(matrix.dom) * index(matrix.cod)
            <= max_entries)

    def cast(self, dtype: type) -> Matrix:
        """
        Cast a matrix to a given ``dtype``, the matrix itself when it
        already has it.

        Parameters:
            dtype : The target datatype.

        Example
        -------
        >>> assert Matrix.id().cast(bool) == Matrix[bool].id()
        >>> m = Matrix[float].id(2)
        >>> assert m.cast(float) is m
        """
        if dtype == self.dtype:
            return self
        return type(self)[dtype](self.array, self.dom, self.cod)

    def promote(self, *others: Matrix) -> type:
        """
        The class of a composite of matrices, i.e. of their product,
        direct sum or sum: the one with the ``dtype`` the backend promotes
        their arrays to, read from the operand that already has it so that
        a declared ``dtype`` survives composition with the integer
        structural matrices, e.g. sympy's ``Expr`` rather than the
        ``numpy.object_`` of its array.

        Parameters:
            others : The matrices to compose with.

        Example
        -------
        >>> m = Matrix([[.5, .5], [.25, .75]], 2, 2)
        >>> assert Matrix.id(2).promote(m) == m.promote(Matrix.id(2))\\
        ...     == Matrix[m.dtype]
        >>> assert Matrix[float].id(2).promote(Matrix.id(2)) == Matrix[float]
        >>> assert Matrix[bool].id(2).promote(Matrix[bool].id(2))\\
        ...     == Matrix[bool]
        """
        with backend() as np:
            dtype = self.array.dtype
            for other in others:
                dtype = np.result_type(
                    np.zeros((), dtype=dtype), other.array)
        if dtype == self.array.dtype:
            return type(self)
        for other in others:
            if dtype == other.array.dtype:
                return type(self)[other.dtype]
        return type(self)[getattr(dtype, "type", dtype)]

    def __class_getitem__(cls, values):
        dtype, = values if isinstance(values, tuple) else (values, )
        if getattr(dtype, "__module__", None) == "numpy":
            with backend('numpy') as np:
                dtype = next((python for python in (bool, int, float, complex)
                              if np.dtype(python).type is dtype), dtype)
        return super().__class_getitem__(dtype)

    def __new__(cls, array, *args, **kwargs):
        if cls.dtype is None:
            _array = asarray(array)
            # The dtype of an np.arrays is a class that contains a type
            # attribute that is the actual type. However, other backends
            # have different structures, so this is the easiest option:
            dtype = getattr(_array.dtype, "type", _array.dtype)
            return cls.__new__(cls[dtype], array, *args, **kwargs)
        return object.__new__(cls)

    def __init__(self, array, dom: Nat | int, cod: Nat | int):
        dom, cod = (Nat(x) if isinstance(x, int) else x for x in (dom, cod))
        assert_isinstance(dom, Nat)
        assert_isinstance(cod, Nat)
        self.dom, self.cod = dom, cod
        with backend() as np:
            array = asarray(array, dtype=self.dtype).reshape(
                (index(dom), index(cod)))
            if isinstance(np, NumPy):
                array = array.view()
                array.flags.writeable = False
        self.array = array

    def __eq__(self, other):
        return isinstance(other, self.ar)\
            and self.dtype == other.dtype\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and (self.array == other.array).all()

    def __le__(self, other):
        """
        The pointwise order on parallel matrices: every entry is at most
        that of the ``other``. A Boolean matrix is a relation and the order
        is inclusion; composition, tensor and transpose are monotone.

        >>> R, S = Matrix[bool]([1, 0, 0, 0], 2, 2), Matrix[bool].id(2)
        >>> assert R <= S and S >= R and not S <= R
        >>> assert R >> R <= S >> S and R @ R <= S @ S
        >>> assert not R <= Matrix[bool]([1, 1], 1, 2)
        >>> assert not Matrix([1, 2], 1, 2) <= Matrix([2, 1], 1, 2)

        Matrices of different data types are not compared:

        >>> R <= R.cast(int)
        Traceback (most recent call last):
        ...
        TypeError: '<=' not supported between instances of 'Matrix[bool]' \
and 'Matrix[int]'

        The complex numbers have no order, as in Python:

        >>> Matrix[complex]([1j], 1, 1) <= Matrix[complex]([2j], 1, 1)
        Traceback (most recent call last):
        ...
        TypeError: Matrices over <class 'complex'> have no pointwise order.
        """
        if not isinstance(other, self.ar):
            return NotImplemented
        if is_complex(self.dtype):
            raise TypeError(messages.MATRIX_ORDER_ERROR.format(self.dtype))
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and (self.array <= other.array).all()

    def is_close(self, other: Matrix, rtol: float = 1.e-8, atol: float = 1.e-8
                 ) -> bool:
        """
        Whether a matrix is numerically close to an ``other``.

        Parameters:
            other : The other matrix with which to check closeness.
            rtol:
                The relative tolerance parameter (see Notes).
                Default value is 1.e-8
            atol :
                The absolute tolerance parameter (see Notes).
                Default value is 1.e-8

        Notes:
        (taken from np.isclose documentation)

            For finite values, isclose uses the following equation to
            test whether two floating point values are equivalent.

            absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

            Unlike the built-in `math.isclose`, the above equation is not
            symmetric in `a` and `b` -- it assumes `b` is the reference
            value -- so that `isclose(a, b)` might be different from
            `isclose(b, a)`.

            Furthermore, the default value of atol is not zero, and is used
            to determine what small values should be considered close to zero.
            The default value is appropriate for expected values of order
            unity: if the expected values are significantly smaller than one,
            it can result in false positives.

            `atol` should be carefully selected for the use case at hand.
            A zero value for `atol` will result in `False` if either `a`
            or `b` is zero.

            `isclose` is not defined for non-numeric data types.
            `bool` is considered a numeric data-type for this purpose

        """
        assert_isinstance(other, type(self))
        assert_isinstance(self, type(other))
        assert_isparallel(self, other)
        with backend() as np:
            return np.isclose(self.array, other.array, rtol, atol).all()

    def __repr__(self):
        np_array = getattr(self.array, 'numpy', lambda: self.array)()
        return type(self).__name__ + f"({array2string(np_array.reshape(-1))},"\
                                     f" dom={self.dom}, cod={self.cod})"

    def to_tree(self) -> dict:
        """
        Serialise a matrix, see :func:`~discopy.utils.dumps`.

        The ``dtype`` travels in the factory name, as it does in the
        ``repr``, so that a matrix comes back with the entries it was
        written with rather than the ones its array implies.

        Example
        -------
        >>> from discopy.utils import dumps, loads
        >>> m = Matrix[float]([1], 1, 1)
        >>> m.to_tree()['factory']
        'matrix.Matrix[float]'
        >>> assert loads(dumps(m)) == m != Matrix([1], 1, 1)
        """
        with backend() as np:
            array = np.array(self.array).reshape(-1).tolist()
        return {
            'factory': factory_name(type(self)),
            'array': array,
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree: dict) -> Matrix:
        """
        Decode a serialised matrix, see :func:`~discopy.utils.loads`.

        Parameters:
            tree : DisCoPy serialisation.

        The class the tree names is the subscripted one, so the pair is
        read through :func:`~discopy.utils.from_tree` rather than off the
        bare class, which would infer a ``dtype`` from the array instead.

        Example
        -------
        >>> import numpy as np
        >>> from discopy.utils import from_tree
        >>> m = Matrix[np.float32]([0, 1, 1, 0], 2, 2)
        >>> assert from_tree(m.to_tree()) == m
        >>> assert Matrix.from_tree(m.to_tree()).dtype == float
        """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        return cls(tree['array'], dom, cod)

    def __iter__(self):
        for i in self.array:
            yield i

    def __bool__(self):
        return bool(self.array)

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    @classmethod
    def id(cls, dom=0) -> Matrix:
        with backend('numpy') as np:
            array = np.identity(index(dom), dtype=int)
        return cls(array, dom, dom)

    twist = id

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        assert_isinstance(other, self.ar)
        assert_iscomposable(self, other)
        cls = self.promote(other)
        left, right = (
            factor.cast(cls.dtype).array for factor in (self, other))
        with backend() as np:
            array = np.matmul(left, right)
        return cls(array, self.dom, other.cod)

    def tensor(self, *others: Matrix):
        """
        The direct sum of ``n`` matrices, i.e. one block-diagonal array with
        a block for each matrix.

        >>> one, two = Matrix([1], 1, 1), Matrix([2], 1, 1)
        >>> assert one.tensor(two) == Matrix([1, 0, 0, 2], 2, 2)
        """
        if not others:
            return self
        for other in others:
            assert_isinstance(other, self.ar)
        cls = self.promote(*others)
        dom = self.dom.tensor(*(other.dom for other in others))
        cod = self.cod.tensor(*(other.cod for other in others))
        blocks, width, j = [], index(cod), 0
        with backend() as np:
            for factor in (self, ) + others:
                rows, columns = index(factor.dom), index(factor.cod)
                left = cls.zero(rows, j).array
                right = cls.zero(rows, width - j - columns).array
                blocks.append(np.concatenate([
                    left, cls(factor.array, factor.dom, factor.cod).array,
                    right], axis=1))
                j += columns
            array = np.concatenate(blocks, axis=0)
        return cls(array, dom, cod)

    def __add__(self, other):
        assert_isinstance(other, self.ar)
        assert_isparallel(self, other)
        return self.promote(other)(
            self.array + other.array, self.dom, self.cod)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    @classmethod
    def zero(cls, dom: Nat | int, cod: Nat | int) -> Matrix:
        """
        Returns the zero matrix of a given shape.

        Examples
        --------
        >>> assert Matrix.zero(2, 2) == Matrix([0, 0, 0, 0], 2, 2)
        """
        with backend() as np:
            return cls(np.zeros(
                (index(dom), index(cod)), dtype=cls.dtype or int), dom, cod)

    @classmethod
    def algebra(cls, monad: Monad, dom: Nat | int, cod: Nat | int) -> Algebra:
        """
        The algebra of a monad on the matrices from ``dom`` to ``cod``, by
        which a functor into matrices extends to the change of base of its
        domain, see :class:`~discopy.enriched.Functor`: the sum of a list of
        matrices, for the :attr:`~discopy.kleisli.monad.List` monad.

        Parameters:
            monad : The monad, the list monad alone here.
            dom : The domain of the matrices.
            cod : The codomain of the matrices.

        Raises:
            TypeError : For any monad but the list monad.

        Example
        -------
        >>> from discopy.kleisli.monad import List
        >>> total = Matrix[int].algebra(List, 1, 2)
        >>> total([Matrix([1, 0], 1, 2), Matrix([2, 3], 1, 2)])
        Matrix[int]([3, 3], dom=Nat(1), cod=Nat(2))
        >>> assert total([]) == Matrix[int].zero(1, 2)

        The algebra is named by its homset, so that two homsets have two:

        >>> total
        monad.algebra.Algebra[List]('sum Nat(1) -> Nat(2)', matrix.Matrix[int])
        >>> assert total != Matrix[int].algebra(List, 2, 1)
        """
        # pylint: disable-next=import-outside-toplevel  # kleisli imports us
        from discopy.kleisli.monad import List
        # pylint: disable-next=import-outside-toplevel  # kleisli imports us
        from discopy.monad import algebra
        if monad != List:
            raise TypeError(messages.NO_ALGEBRA.format(
                factory_name(cls), monad))
        dom, cod = map(cls.ob.cast, (dom, cod))
        return algebra.Algebra[List](  # ty: ignore[invalid-type-form]
            f"sum {dom} -> {cod}", cls,
            lambda matrices: sum(matrices, cls.zero(dom, cod)))

    @classmethod
    def swap(cls, left: Nat | int, right: Nat | int) -> Matrix:
        """
        The matrix that swaps left and right dimensions.

        Parameters:
            left : The left dimension.
            right : The right dimension.

        Example
        -------
        >>> Matrix.swap(1, 1)
        Matrix[int]([0, 1, 1, 0], dom=Nat(2), cod=Nat(2))
        >>> Matrix.swap(2,1)
        Matrix[int]([0, 1, 0, 0, 0, 1, 1, 0, 0], dom=Nat(3), cod=Nat(3))
        """
        return cls.permutation([1, 0], [left, right])

    braid = swap

    @classmethod
    def permutation(
            cls, xs: Sequence[int],
            doms: Sequence[Nat | int] | None = None) -> Matrix:
        """
        The matrix that permutes blocks of dimensions.

        Overrides the default of :class:`~discopy.abc.SymmetricCategory`, which
        composes ``len(xs)`` swaps, by writing each block in one pass.

        Parameters:
            xs : The permutation, as a list of indices.
            doms : The dimension of each block, one by default.

        When every block is a single dimension, the matrix is the permutation
        matrix itself, i.e. ``array[i][j]`` iff ``xs[j] == i``.

        Example
        -------
        >>> assert Matrix.permutation([1, 0]) == Matrix.swap(1, 1)
        >>> Matrix.permutation([1, 2, 0])
        Matrix[int]([0, 0, 1, 1, 0, 0, 0, 1, 0], dom=Nat(3), cod=Nat(3))
        >>> ones = Matrix.permutation([1, 2, 0], [1, 1, 1])
        >>> assert ones == Matrix.permutation([1, 2, 0])

        Blocks of more than one dimension move as a whole.

        >>> assert Matrix.permutation([1, 0], [1, 2]) == Matrix.swap(1, 2)
        """
        xs = list(xs)
        doms = len(xs) * [1] if doms is None else list(map(index, doms))
        if list(range(len(doms))) != sorted(xs) or min(doms, default=0) < 0:
            raise ValueError
        offsets = [sum(doms[:i]) for i in range(len(doms))]
        sources = [offsets[i] + j for i in xs for j in range(doms[i])]
        with backend() as np:
            array = np.array([
                [int(i == j) for j in sources] for i in range(len(sources))],
                dtype=cls.dtype or int)
        return cls(array, len(sources), len(sources))

    def transpose(self) -> Matrix:
        with backend() as np:
            array = np.moveaxis(self.array, 0, 1)
        return type(self)(array, self.cod, self.dom)

    def conjugate(self) -> Matrix:
        with backend() as np:
            array = np.conj(self.array)
        return type(self)(array, self.dom, self.cod)

    def dagger(self) -> Matrix:
        return self.conjugate().transpose()

    def map(self, func: Callable, dtype: type | None = None) -> Matrix:
        """
        Apply a function to each entry of a matrix.

        Parameters:
            func : The function to apply.
            dtype : The ``dtype`` of the result, by default the one
                :meth:`promote` gives for the matrix and the entries
                computed, so that a map widens and never truncates.

        Example
        -------
        >>> Matrix([1, 2], 1, 2).map(lambda x: x + 0.5)
        Matrix[float]([1.5, 2.5], dom=Nat(1), cod=Nat(2))
        >>> Matrix([1, 2], 1, 2).map(lambda x: x + 0.5, dtype=int)
        Matrix[int]([1, 2], dom=Nat(1), cod=Nat(2))
        """
        array = list(map(func, self.array.reshape(-1)))
        if dtype is None and array:
            dtype = self.promote(Matrix(array, 1, len(array))).dtype
        return type(self)[dtype or self.dtype](array, self.dom, self.cod)

    def round(self, decimals=0) -> Matrix:
        """ Rounds the entries of a matrix up to a number of decimals. """
        with backend() as np:
            array = np.round(self.array, decimals=decimals)
        return type(self)(array, self.dom, self.cod)

    @classmethod
    def copy(cls, x: Nat, n: int = 2) -> Matrix:
        """
        The diagonal ``x -> n * x`` of the direct sum, i.e. ``n`` identities
        side by side: ``array[i, j]`` is one iff ``j % x == i``.

        Parameters:
            x : The dimension to copy.
            n : The number of copies.

        Example
        -------
        >>> Matrix.copy(2)
        Matrix[int]([1, 0, 1, 0, 0, 1, 0, 1], dom=Nat(2), cod=Nat(4))
        >>> assert Matrix.copy(2) >> Matrix.id(2) @ Matrix.discard(2)\\
        ...     == Matrix.id(2)
        """
        dim, n = index(x), index(n)
        with backend() as np:
            array = np.array([
                [int(j % dim == i) for j in range(n * dim)]
                for i in range(dim)],
                dtype=cls.dtype or int)
        return cls(array, dim, n * dim)

    @classmethod
    def merge(cls, x: Nat, n: int = 2) -> Matrix:
        """ The codiagonal ``n * x -> x``, the dagger of :meth:`copy`. """
        return cls.copy(x, n).dagger()

    @classmethod
    def ones(cls, x: Nat | int) -> Matrix:
        """ The unit ``0 -> x`` of the direct sum, an alias of :meth:`noise`.
        """
        return cls.noise(x)

    @classmethod
    def basis(cls, x: Nat | int, i: int) -> Matrix:
        """
        The ``i``-th basis vector of dimension ``x``.

        Parameters:
            x : The dimension of the basis vector.
            i : The index of the basis vector.

        Example
        -------
        >>> Matrix.basis(4, 2)
        Matrix[int]([0, 0, 1, 0], dom=Nat(1), cod=Nat(4))
        >>> assert Matrix.basis(0, 0).dtype == Matrix.id(0).dtype
        """
        with backend() as np:
            array = np.array([[int(i == j) for j in range(index(x))]],
                             dtype=cls.dtype or int)
        return cls(array, 1, x)

    def repeat(self) -> Matrix:
        """
        The reflexive transitive closure of a boolean matrix.

        Example
        -------
        >>> Matrix[bool]([0, 1, 1, 0], 2, 2).repeat()
        Matrix[bool]([True, True, True, True], dom=Nat(2), cod=Nat(2))
        """
        if self.dtype != bool or self.dom != self.cod:
            raise TypeError(messages.MATRIX_REPEAT_ERROR)
        result, length = self.id(self.dom) + self, 1
        while length < index(self.dom):
            result, length = result >> result, 2 * length
        return result

    def resolvent(self) -> Matrix:
        """
        The inverse of ``id - self`` over a field, see :func:`is_field`, i.e.
        the sum of its powers where it converges, raising ``TypeError`` over
        integers, ``ValueError`` when singular unless traced by ``jax.jit``.

        Example
        -------
        >>> Matrix[float]([0, 1, 0, 0], 2, 2).resolvent()
        Matrix[float]([1., 1., 0., 1.], dom=Nat(2), cod=Nat(2))
        >>> Matrix[float]([1], 1, 1).resolvent()
        Traceback (most recent call last):
            ...
        ValueError: id - D is singular: the trace over a field is partial.
        """
        if self.dom != self.cod or not is_field(self.dtype):
            raise TypeError(messages.MATRIX_RESOLVENT_ERROR)
        square = self.array.reshape(2 * (isqrt(prod(self.array.shape)), ))
        with backend() as np:
            try:
                inverse = np.linalg.inv(np.eye(len(square)) - square)
            except getattr(np.linalg, "LinAlgError", ()) as error:
                raise ValueError(messages.MATRIX_SINGULAR) from error
            try:
                singular = not np.isfinite(inverse).all()
            except TypeError:
                singular = False
            if singular:
                raise ValueError(messages.MATRIX_SINGULAR)
            return type(self)(inverse, self.dom, self.cod)

    def delay(self, n_steps: int = 1) -> Matrix:
        """
        The delay of a matrix is itself, as the twist of a compact category
        is the identity: a matrix has no time steps, so a functor sends a
        delayed box to the image of the box.

        >>> f = Matrix[int]([[1, 2]], 1, 2)
        >>> assert f.delay() == f == f.delay(3)
        """
        del n_steps
        return self

    def trace(self, n=1, left=False) -> Matrix:
        """
        The partial trace ``A + B >> D* >> C`` of a matrix with blocks
        ``A: dom -> cod``, ``B: dom -> n``, ``C: n -> cod``, ``D: n -> n``,
        with ``D*`` the :meth:`repeat` over ``bool`` and the
        :meth:`resolvent` over a field: the trace of Haghverdi and Scott,
        defined when ``id - D`` is invertible, see :cite:t:`MalherbeEtAl11`,
        Definition 3.11, and the sum over unrollings when ``D`` is nilpotent.

        Parameters:
            n : The number of dimensions to trace.
            left : Whether to trace the first ``n`` dimensions rather than
                the last, by conjugating with swaps.

        Example
        -------
        >>> assert Matrix[bool].swap(1, 1).trace() == Matrix[bool].id(1)
        >>> f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
        >>> assert f.trace() == Matrix[bool]([[1]], 1, 1)
        >>> assert f.trace(left=True) == Matrix[bool]([[0]], 1, 1)
        >>> g = Matrix[float]([[0.5, 1], [1, 0.25]], 2, 2)
        >>> assert g.trace().is_close(Matrix[float]([[0.5 + 4 / 3]], 1, 1))
        """
        dom, cod = index(self.dom) - n, index(self.cod) - n
        if left:
            return (self.swap(dom, n) >> self >> self.swap(n, cod)).trace(n)
        A, B, C, D = (row >> self >> column
                      for row in [self.id(dom) @ self.ones(n),
                                  self.ones(dom) @ self.id(n)]
                      for column in [self.id(cod) @ self.discard(n),
                                     self.discard(cod) @ self.id(n)])
        star = D.repeat() if self.dtype == bool else D.resolvent()
        return A + (B >> star >> C)

    def lambdify(
            self, *symbols: "sympy.Symbol", dtype=None, **kwargs) -> Callable:
        from sympy import lambdify
        with backend() as np:
            modules = kwargs.pop("modules", np.module)
            array = lambdify(
                symbols, self.array.tolist(), modules=modules, **kwargs)
        dtype = dtype or self.dtype
        return lambda *xs: type(self)[dtype](array(*xs), self.dom, self.cod)

    def subs(self, *args) -> Matrix:
        return self.map(lambda x: getattr(x, "subs", lambda y, *_: y)(*args))

    def grad(self, var, **params) -> Matrix:
        """ Gradient with respect to variables. """
        return self.map(lambda x:
                        getattr(x, "diff", lambda _: 0)(var, **params))


def asarray(entries, dtype=None):
    """
    The array of the active :func:`backend` holding nested ``entries``,
    stacking the ones the backend already owns so that a PyTorch tensor or
    a JAX tracer keeps its gradient: converting such an entry one by one
    reads it as a scalar and detaches it, wherever among the entries it
    sits. A backend whose arrays are numpy's needs no stacking, and
    ``asarray`` leaves its array alone.

    Parameters:
        entries : The array, or the nested lists of its entries.
        dtype : The ``dtype`` of the result, the one of the entries by
            default.

    Example
    -------
    >>> import numpy
    >>> assert (asarray([[0, 1], [1, 0]]) == numpy.eye(2)[::-1]).all()
    >>> with backend('pytorch'):  # doctest: +EXTRA
    ...     x = asarray([0.5])
    ...     assert asarray(x) is x
    """
    def stack(entries, np):
        if isinstance(entries, (list, tuple)):
            return np.stack([stack(x, np) for x in entries])
        return np.array(entries, dtype=dtype)

    def leaves(entries):
        if isinstance(entries, (list, tuple)):
            for entry in entries:
                yield from leaves(entry)
        else:
            yield entries

    with backend() as np:
        array = type(np.array(0))
        if array.__module__ != "numpy" and any(
                isinstance(entry, array) for entry in leaves(entries)):
            return stack(entries, np)
        return np.asarray(entries, dtype=dtype)


def array2string(array, **params):
    """
    Numpy array pretty print, eliding past ``config.NUMPY_THRESHOLD`` entries
    without touching numpy's global print options, with ``nan`` and ``inf``
    spelled by :func:`spell_nonfinite` so that ``eval`` reads them back.

    >>> import numpy
    >>> threshold = numpy.get_printoptions()['threshold']
    >>> array2string(numpy.arange(3))
    '[0, 1, 2]'
    >>> array2string(numpy.array([1., numpy.nan]))
    "[1., float('nan')]"
    >>> assert numpy.get_printoptions()['threshold'] == threshold
    """
    import numpy
    array = numpy.asarray(array)
    params = dict({'threshold': config.NUMPY_THRESHOLD,
                   'formatter': {'object': repr_nonfinite}}, **params)
    string = numpy.array2string(
        array, **dict(
            params, separator=', '))  # ty: ignore[invalid-argument-type]
    string = string.replace('[ ', '[').replace('  ', ' ')
    return spell_nonfinite(string) if array.dtype.kind in 'fc' else string


NONFINITE = re.compile(r"\b(nan|inf)(j?)\b")


def spell_nonfinite(string: str) -> str:
    """
    Spell the ``nan`` and ``inf`` that numpy prints for an array of floats or
    complex numbers as ``float('nan')`` and ``float('inf')``, and ``nanj``
    and ``infj`` as ``complex('nanj')`` and ``complex('infj')``, so that
    ``eval`` reads them back with no import. The string holds numbers only:
    an entry of any other type is spelled by :func:`repr_nonfinite`.

    >>> spell_nonfinite("[nan, -inf, 1.+nanj]")
    "[float('nan'), -float('inf'), 1.+complex('nanj')]"
    """
    def spell(match):
        token, imaginary = match.groups()
        return f"complex('{token}j')" if imaginary else f"float('{token}')"
    return NONFINITE.sub(spell, string)


def repr_nonfinite(value) -> str:
    """
    The ``repr`` of a value or of nested lists and tuples of values, with
    each float or complex number that is not finite spelled as
    ``float('nan')`` or ``complex('(1+nanj)')``, so that ``eval`` reads it
    back with no import.
    Any other value is its ``repr``, e.g. a symbol named ``inf`` stays one.

    >>> import sympy
    >>> nan, inf = float('nan'), float('inf')
    >>> repr_nonfinite([nan, [-inf, complex(1, nan)]])
    "[float('nan'), [float('-inf'), complex('(1+nanj)')]]"
    >>> repr_nonfinite([sympy.Symbol('inf'), 1.0])
    '[inf, 1.0]'
    >>> repr_nonfinite((nan, ))
    "(float('nan'),)"
    """
    if isinstance(value, list):
        return "[" + ", ".join(map(repr_nonfinite, value)) + "]"
    if isinstance(value, tuple):
        entries = ", ".join(map(repr_nonfinite, value))
        return f"({entries},)" if len(value) == 1 else f"({entries})"
    if isinstance(value, float) and not isfinite(value):
        return f"float('{value}')"
    if isinstance(value, complex) and not cmath.isfinite(value):
        return f"complex('{value}')"
    return repr(value)


class Backend:
    """
    A matrix backend.

    Parameters:
        module : The main module of the backend.
        array : The array class of the backend.
    """
    def __init__(self, module: ModuleType, array: Callable | None = None):
        self.module, self.array = module, array or module.array

    def __getattr__(self, attr):
        return getattr(self.module, attr)

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __eq__(self, other):
        return type(self) is type(other)\
            and getattr(self, "base", None) == getattr(other, "base", None)

    def __hash__(self):
        return hash((factory_key(type(self)), getattr(self, "base", None)))


class NumPy(Backend):
    """ NumPy backend. """
    def __init__(self):
        import numpy
        super().__init__(numpy)


class JAX(Backend):
    """ JAX backend. """
    def __init__(self):
        import jax
        super().__init__(jax.numpy)


class PyTorch(Backend):
    """ PyTorch backend. """
    def __init__(self):
        import torch
        super().__init__(torch, array=torch.as_tensor)


class TensorFlow(Backend):
    """ TensorFlow backend. """
    def __init__(self):
        import tensorflow.experimental.numpy as tnp  # ty: ignore
        from tensorflow.python.ops.numpy_ops import np_config  # ty: ignore
        np_config.enable_numpy_behavior()
        super().__init__(tnp)


class Contractor(Backend):
    """
    A contraction engine, i.e. a backend whose ``einsum`` contracts the
    arrays of another backend, to which it delegates everything else, so
    that e.g. gradients of pytorch arrays survive the contraction.

    A plain :class:`Backend` contracts with its own ``einsum``.

    Parameters:
        base : The backend of the arrays, the active one at contraction
            time by default.

    Example
    -------
    >>> from discopy.tensor import Box, Dim
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> diagram = vector >> vector[::-1]
    >>> assert diagram.eval(contractor=OptEinsum()) == diagram.eval()
    >>> OptEinsum(), Quimb(NumPy())
    (OptEinsum(), Quimb(NumPy()))
    >>> assert eval(repr(Quimb(NumPy()))) == Quimb(NumPy()) != Quimb()
    """
    def __init__(self, base: Backend | None = None):
        # pylint: disable=super-init-not-called
        # the module and array of the base are read at contraction time
        self.base = base

    @property
    def module(self) -> ModuleType:
        """ The module of the base, the active backend by default. """
        return (self.base or get_backend()).module

    @property
    def array(self) -> Callable:
        """ The array of the base, the active backend by default. """
        return (self.base or get_backend()).array

    def __repr__(self):
        base = "" if self.base is None else repr(self.base)
        return f"{type(self).__name__}({base})"

    def einsum(self, *operands, **params):
        """
        Contract interleaved operands, i.e. arrays each followed by its
        list of integer indices, then the list of output indices.

        Parameters:
            operands : The interleaved operands and the output indices.
            params : The parameters of the engine, e.g. ``optimize``.
        """
        raise NotImplementedError


class OptEinsum(Contractor):
    """ The contraction engine of the optional ``opt_einsum`` package. """
    def einsum(self, *operands, **params):
        try:
            import opt_einsum
        except ImportError as error:
            if error.name != "opt_einsum":
                raise
            raise ImportError(
                "Contracting more than "
                f"{config.MAX_EINSUM_INDICES} indices requires "
                "opt_einsum, run `pip install opt_einsum`.") from error
        return opt_einsum.contract(*operands, **params)


class Quimb(Contractor):
    """
    The contraction engine of the optional ``quimb`` package, where
    ``optimize`` may be a ``cotengra`` path optimizer, and a ``max_bond``
    parameter or a compressed optimizer selects approximate contraction.

    Gradients of jax and pytorch arrays survive the contraction with
    ``autoray >= 0.9``, whose older versions fall back to numpy when
    reusing a cached contraction tree.
    """
    def network(self, arrays: list, indices: list[list[int]],
                names: dict[int, str] | None = None
                ) -> "quimb.tensor.TensorNetwork":
        """
        The quimb tensor network of arrays with integer indices, each
        index named ``names[j]`` or ``f"w{j}"``. An index repeated on one
        array, i.e. a trace, is summed over beforehand.

        Parameters:
            arrays : The arrays.
            indices : The list of integer indices of each array.
            names : The names of the indices, ``f"w{j}"`` by default.
        """
        import quimb.tensor as qtn
        names = names or {}
        tensors = []
        for array, inds in zip(arrays, indices):
            kept = [j for j in inds if inds.count(j) == 1]
            if kept != inds:
                local = {j: i for i, j in enumerate(dict.fromkeys(inds))}
                array = self.module.einsum(
                    array, [local[j] for j in inds],
                    [local[j] for j in kept])
            tensors.append(qtn.Tensor(
                array, inds=tuple(names.get(j, f"w{j}") for j in kept)))
        return qtn.TensorNetwork(tensors)

    def einsum(self, *operands, optimize="greedy", **params):
        import quimb.tensor as qtn
        *operands, output = operands
        network = self.network(operands[::2], operands[1::2])
        compressed = "max_bond" in params\
            or getattr(optimize, "compressed", False)\
            or "Compressed" in type(optimize).__name__
        for tensor in network.tensors if compressed else ():
            if getattr(tensor.data.dtype, "kind", "") in "?bui":
                tensor.modify(data=tensor.data.astype("complex128"))
        method = network.contract_compressed if compressed\
            else network.contract
        result = method(output_inds=[f"w{j}" for j in output],
                        optimize=optimize, **params)
        return result.data if isinstance(result, qtn.Tensor) else result


def is_field(dtype) -> bool:
    """ Whether a ``dtype`` is floating-point or complex, in any backend. """
    flags = ("is_floating_point", "is_floating", "is_complex")
    with backend("numpy") as numpy:
        return any(getattr(dtype, flag, False) is True for flag in flags)\
            or getattr(dtype, "kind", None) in ("f", "c")\
            or isinstance(dtype, type)\
            and issubclass(dtype, (float, complex, numpy.inexact))


def is_complex(dtype) -> bool:
    """ Whether a ``dtype`` is complex, in any backend. """
    with backend("numpy") as numpy:
        return getattr(dtype, "is_complex", False) is True\
            or getattr(dtype, "kind", None) == "c"\
            or isinstance(dtype, type)\
            and issubclass(dtype, (complex, numpy.complexfloating))


BACKENDS = {
    'numpy': NumPy,
    'jax': JAX,
    'pytorch': PyTorch,
    'tensorflow': TensorFlow,
}

BackendName = Literal["numpy", "jax", "pytorch", "tensorflow"]


@contextmanager
def backend(name: BackendName | None = None,
            _stack: list = [config.DEFAULT_BACKEND], _cache=dict()):
    """
    Context manager for matrix backend.

    Parameters:
        name : The name of the backend, default is ``"numpy"``.

    Example
    -------
    >>> with backend('jax'):  # doctest: +EXTRA
    ...     assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...         == 'jaxlib._jax'
    """
    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()


def set_backend(name: BackendName) -> None:
    """
    Override the default backend.

    Parameters:
        name : The name of the backend.

    Example
    -------
    >>> set_backend('jax')
    >>> assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...     == 'jaxlib._jax'  # doctest: +EXTRA
    >>> set_backend('numpy')
    >>> assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...     == 'numpy'
    """
    backend.__wrapped__.__defaults__[1][-1] = name


def get_backend() -> Backend:
    """
    Get the current backend.

    Example
    -------
    >>> set_backend('jax')
    >>> assert isinstance(get_backend(), JAX)  # doctest: +EXTRA
    >>> set_backend('numpy')
    >>> assert isinstance(get_backend(), NumPy)
    """
    with backend() as result:
        return result
