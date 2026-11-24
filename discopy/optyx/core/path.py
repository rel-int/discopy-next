"""

Overview
--------

The category :class:`Matrix` and the syntax :class:`Diagram`
of matrices with creations and post-selections. The module
supports representing the :class:`lo` fragment
of Optyx diagrams as matrices. It enables the
computation of the amplitudes and probabilities
of the diagrams by evaluting
permanents of underlying matrices (either directly
or via Perceval [FGL+23]_). The :code:`to_path` method
of Optyx diagrams which belong to the W fragment of the
:class:`zw` calculus (including :class:`lo` circuits,
n-photon states and effects)
returns a :class:`Matrix` object.

Classes
-------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Sum
    Amplitudes
    Probabilities


Functions
-------------

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        npperm

Examples of usage
------------------

**Hong-Ou-Mandel effect**

We can check the Hong-Ou-Mandel effect by
evaluating the permanent of the underlying matrix:

>>> from discopy.optyx.core.zw import Create, Select, Split, Merge, Id
>>> from discopy.optyx.photonic import BS
>>> HOM = Create(1, 1) >> BS.get_kraus()
>>> assert np.allclose(HOM.to_path().eval().array,\\
... Amplitudes([0.+0.70710678j, -0.+0.j    , 0.+0.70710678j],\\
... dom=1, cod=3).array)

>>> assert np.allclose(HOM.to_path().prob().array, \\
... Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], \\
... dom=1, cod=3).array)
>>> left = Create(1, 1) >> BS.get_kraus() >> Select(2, 0)
>>> left.to_path().prob()
Probabilities[complex]([0.5+0.j], dom=Nat(1), cod=Nat(1))

We can also show the Hong-Ou-Mandel effect by
using the rules of the :class:`path` calculus:

>>> from discopy.optyx.core.zw import W, Endo, SWAP, Create, Select
>>> left_hs = (Create(1, 1) >> \\
... W(2) @ W(2) >> \\
... Endo(1j) @ Id(1) @ Id(1) @ Endo(1j) >> \\
... Id(1) @ SWAP @ Id(1) >> \\
... W(2).dagger() @ W(2).dagger() >> \\
... Select(1, 1))
>>> left_hs.draw(doctest="docs/_static/optyx/left_hs.png")

.. image:: /_static/optyx/left_hs.png
    :align: center

>>> left_hs.to_path().prob_with_perceval().array[0, 0]
np.complex128(0j)

According to the rewrite rules of the :class:`path` calculus,
this should be equal to:

>>> right_hs = ((Create(1, 1) >> \\
... Endo(1j) @ Endo(1j) >> \\
... Select(1, 1)) +
... (Create(1, 1) >> \\
... SWAP >> \\
... Select(1, 1)))
>>> right_hs.draw(doctest="docs/_static/optyx/right_hs.png")

.. image:: /_static/optyx/right_hs.png

Let us now evaluate this using :code:`DisCoPy` tensor:

>>> right_hs.to_tensor().eval().array
array(0.+0.j)

**Bell state**

We can construct a Bell state in dual rail encoding:

>>> plus = Create() >> Split(2)
>>> state = plus >> Id(1) @ plus @ Id(1)
>>> bell = state @ state \\
...     >> Id(2) @ (BS.get_kraus() @ \\
...     BS.get_kraus().dagger() >> state.dagger()) @ Id(2)
>>> H, V = Select(1, 0), Select(0, 1)
>>> assert np.allclose(
...     (bell >> H @ H).to_path().eval().array,
...     (bell >> V @ V).to_path().eval().array)
>>> assert np.allclose(
...     (bell >> V @ H).to_path().eval().array,
...     (bell >> H @ V).to_path().eval().array)

**Number operator**

We can define the number operator and compute its expectation.

>>> num_op = Split(2) >> Id(1) @ Select(1) >> Id(1) @ Create(1) >> Merge(2)
>>> expectation = lambda n: Create(n) >> num_op >> Select(n)
>>> assert np.allclose(expectation(5).to_path().eval().array, np.array([5.]))

"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING, Callable, ClassVar
from operator import index

import numpy as np

from discopy import sum as formal_sum
from discopy.abc import Nat
from discopy.cat import assert_iscomposable
from discopy.utils import (
    AxiomError, factory, factory_key, factory_name, from_tree,
    resolve_factory, unbiased)
import discopy.matrix as underlying
from discopy.tensor import Tensor
from discopy.optyx.utils.misc import occupation_numbers, amplitudes_2_tensor

if TYPE_CHECKING:
    import perceval as pcvl


def npperm(matrix):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151.
    """
    n = matrix.shape[0]
    if not n:
        return 1
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = matrix.sum(axis=0)
    p = np.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * matrix[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


@factory
class Matrix(underlying.Matrix):
    """
    Matrix with photon creations and post-selections,
    evaluated as :class:`Amplitudes`.

    Parameters:
        array : underlying array
        dom : number of modes in, an int or a :class:`Nat`
        cod : number of modes out, an int or a :class:`Nat`
        creations : list of occupation numbers
        selections : list of occupation numbers
        normalisation : normalisation factor dependent on number of photons.
        scalar : global scalar independent of number of photons

    Example
    -------
    >>> from discopy.optyx.core.zw import Split, Select, Create, Merge, Id
    >>> array = np.array([[1, 1], [1, 0]])
    >>> matrix = Matrix(array, 1, 1, creations=(1,), selections=(1,))
    >>> matrix.eval(3)
    Amplitudes([3.+0.j], dom=Nat(1), cod=Nat(1))
    >>> num_op = Split(2) >> Select() @ Id(1) >> Create() @ Id(1) >> Merge(2)
    >>> assert np.allclose(num_op.to_path().eval(4).array,
    ...                    matrix.eval(4).array)

    Note
    ----
    Matrices add formally, see :class:`Sum`.
    """

    dtype = complex
    sum_factory: ClassVar[type["Sum"]]

    def __new__(
        cls,
        array,
        dom,
        cod,
        creations=(),
        selections=(),
        normalisation=1,
        scalar=1,
    ):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def __init__(
        self,
        array,
        dom: int | Nat,
        cod: int | Nat,
        creations: tuple[int, ...] = (),
        selections: tuple[int, ...] = (),
        normalisation=1,
        scalar=1,
    ):
        dom, cod = index(dom), index(cod)
        self.udom, self.ucod = dom + len(creations), cod + len(selections)
        super().__init__(array, self.udom, self.ucod)
        self.dom, self.cod = Nat(dom), Nat(cod)
        self.creations, self.selections = creations, selections
        self.normalisation = normalisation
        self.scalar = scalar

    def __eq__(self, other):
        def fields(matrix):
            return (tuple(matrix.creations), tuple(matrix.selections),
                    matrix.normalisation, matrix.scalar)
        return isinstance(other, self.ar) \
            and fields(self) == fields(other) and super().__eq__(other)

    @classmethod
    def zero(cls, dom: int | Nat, cod: int | Nat) -> Sum:
        """The formal zero, i.e. the empty :class:`Sum` over ``cls.dtype``."""
        return cls.sum_factory((), dom, cod, cls.dtype)

    def __add__(self, other: Matrix | Sum) -> Sum:
        return self.sum_factory((self, )) + other

    @property
    def net_photons(self) -> int:
        """
        The photons a matrix adds, its creations less its selections: it
        sends the ``n``-photon sector to the ``n + net_photons`` one.
        """
        return sum(self.creations) - sum(self.selections)

    @property
    def umatrix(self) -> underlying.Matrix:
        """
        Underlying matrix with `len(creations) + dom` inputs and
        `len(selections) + cod` outputs.
        """
        return underlying.Matrix[self.dtype](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Matrix | Sum) -> Matrix | Sum:
        """Sequential composition of QPath matrices"""
        if isinstance(other, Sum):
            return self.sum_factory((self, )).then(other)
        assert_iscomposable(self, other)
        M = underlying.Matrix[self.dtype]
        left, right = len(self.selections), len(other.creations)
        umatrix = (
            self.umatrix @ right
            >> self.cod @ M.swap(left, right)
            >> other.umatrix @ left
        )
        creations = self.creations + other.creations
        selections = other.selections + self.selections
        scalar = self.scalar * other.scalar
        normalisation = self.normalisation * other.normalisation
        return Matrix[self.dtype](
            umatrix.array,
            self.dom,
            other.cod,
            creations,
            selections,
            normalisation,
            scalar,
        )

    @unbiased
    def tensor(self, other: Matrix | Sum) -> Matrix | Sum:
        """Parallel composition of QPath matrices"""
        if isinstance(other, Sum):
            return self.sum_factory((self, )).tensor(other)
        M = underlying.Matrix[self.dtype]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.selections), len(other.selections)
        umatrix = (
            self.dom @ M.swap(other.dom, a) @ b
            >> self.umatrix @ other.umatrix
            >> self.cod @ M.swap(c, other.cod) @ d
        )
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        creations = self.creations + other.creations
        selections = self.selections + other.selections
        normalisation = self.normalisation * other.normalisation
        scalar = self.scalar * other.scalar
        return Matrix[self.dtype](
            umatrix.array,
            dom,
            cod,
            creations,
            selections,
            normalisation,
            scalar,
        )

    def dagger(self) -> Matrix:
        """Adjoint QPath matrix"""
        array = self.umatrix.dagger().array
        return Matrix[self.dtype](
            array,
            self.cod,
            self.dom,
            self.selections,
            self.creations,
            self.normalisation.conjugate(),
            self.scalar.conjugate(),
        )

    def __repr__(self):
        return (
            super().__repr__()[:-1]
            + f", creations={self.creations}"
            + f", selections={self.selections}"
            + f", normalisation={self.normalisation}"
            + f", scalar={self.scalar})"
        )

    def to_tree(self) -> dict:
        """
        The tree names the whole profile, as the representation does.

        The ``array`` a matrix writes is its underlying one, with
        ``len(creations)`` rows and ``len(selections)`` columns more than
        its ``dom`` and ``cod``, so a tree without them cannot be read
        back at all, see :meth:`from_tree`.

        Example
        -------
        >>> from discopy.utils import dumps, loads
        >>> matrix = Matrix(np.array([[1, 1], [1, 0]]), 1, 1,
        ...                 creations=(1,), selections=(1,), scalar=2j)
        >>> assert loads(dumps(matrix)) == matrix
        """
        return dict(
            super().to_tree(),
            creations=self.creations,
            selections=self.selections,
            normalisation=self.normalisation,
            scalar=self.scalar)

    @classmethod
    def from_tree(cls, tree: dict) -> Matrix:
        """
        Reads the whole profile back, see :meth:`to_tree`.

        The ``dom`` and ``cod`` of the tree are mode counts, which the
        constructor turns into the shape of the array together with the
        creations and the selections, so ``super().from_tree`` would
        reshape the underlying array into the modes alone.

        Each field of the profile falls back to the default the
        constructor gives it, so a tree written before they travelled —
        which is one with no creation and no selection, the only kind that
        could be read back at all — still loads.
        """
        return cls(
            tree['array'],
            from_tree(tree['dom']),
            from_tree(tree['cod']),
            tuple(tree.get('creations', ())),
            tuple(tree.get('selections', ())),
            tree.get('normalisation', 1),
            tree.get('scalar', 1))

    def dilate(self) -> Matrix:
        """
        Returns an equivalent :class:`Matrix` with unitary underlying matrix.

        Example
        -------
        >>> from discopy.optyx.core.zw import Split, Select, Create, Merge, Id
        >>> num_op = Split(2) >> Select() @ Id(1) \\
        ...           >> Create() @ Id(1) >> Merge(2)
        >>> U = num_op.to_path().dilate()
        >>> assert np.allclose(
        ...     (U.umatrix >> U.umatrix.dagger()).array, np.eye(4))
        >>> assert np.allclose(U.eval(5).array, num_op.to_path().eval(5).array)
        """
        dom, cod = self.umatrix.dom, self.umatrix.cod
        A = self.umatrix.array
        U, S, Vh = np.linalg.svd(A)
        s = max(S) if max(S) > 1 else 1
        defect0 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2),
             [1 for _ in range(index(dom) - len(S))]]
        )
        defect1 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2),
             [1 for _ in range(index(cod) - len(S))]]
        )
        defect_left = U.dot(np.diag(defect0)).dot(U.conj().T)
        defect_right = (Vh.conj().T).dot(np.diag(defect1)).dot(Vh)
        unitary = np.block(
            [[A / s, defect_left], [defect_right, -A.conj().T / s]]
        )
        creations = self.creations + index(cod) * (0,)
        selections = self.selections + index(dom) * (0,)

        return Matrix(
            unitary,
            self.dom,
            self.cod,
            creations,
            selections,
            normalisation=s * self.normalisation,
            scalar=self.scalar,
        )

    def eval(
        self, n_photons=0, permanent=npperm, as_tensor=False
    ) -> Amplitudes:
        """Evaluates the :class:`Amplitudes` of a the QPath matrix"""
        dom_basis = occupation_numbers(n_photons, index(self.dom))
        n_photons_out = n_photons + self.net_photons
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        cod_basis = occupation_numbers(n_photons_out, index(self.cod))

        array = np.zeros((len(dom_basis), len(cod_basis)),
                         dtype=self.dtype or int)
        normalisation = self.normalisation ** (n_photons + sum(self.creations))
        for i, open_creations in enumerate(dom_basis):
            for j, open_selections in enumerate(cod_basis):
                creations = open_creations + self.creations
                selections = open_selections + self.selections
                matrix = self.array[np.ix_(
                    [m for m, n in enumerate(creations) for _ in range(n)],
                    [m for m, n in enumerate(selections) for _ in range(n)])]
                divisor = np.sqrt(
                    np.prod([factorial(n) for n in creations + selections])
                )
                val = self.scalar * normalisation * permanent(matrix) / divisor
                array[i, j] = val
        if as_tensor:
            return amplitudes_2_tensor(array, dom_basis, cod_basis)
        return Amplitudes[self.dtype](array, len(dom_basis), len(cod_basis))

    def prob(
        self,
        n_photons=0,
        permanent=npperm,
        with_perceval=False,
        as_tensor=False,
    ) -> Probabilities:
        """Computes the Born rule of the amplitudes of the :class:`Matrix`"""
        if with_perceval:
            return self.prob_with_perceval(n_photons, as_tensor=as_tensor)
        amplitudes = self.eval(n_photons, permanent, as_tensor)
        probabilities = np.abs(amplitudes.array) ** 2
        if as_tensor:
            return Tensor(probabilities, amplitudes.dom, amplitudes.cod)
        return Probabilities[self.dtype](
            probabilities, amplitudes.dom, amplitudes.cod
        )

    def prob_with_perceval(
        self, n_photons=0, simulator: str = "SLOS", as_tensor=False
    ) -> Probabilities:
        """
        Computes the Born rule of the amplitudes of the :class:`Matrix` using
        the perceval library

        Note
        ----
        If the :class:`Matrix` is non-unitary, first :meth:`dilate` is called
        to create a unitary, whose normalisation is then put back on the
        probabilities. The selections pick out the outcomes that satisfy them
        rather than being given to perceval as a post-selection, which
        renormalises over the outcomes that survive it and so answers a
        conditional probability.

        Example
        -------
        >>> import numpy as np
        >>> from discopy.optyx.core.zw import (
        ...   Split, Select, Create,
        ...   Merge, Id, Endo, SWAP
        ... )
        >>> theta, phi = np.pi / 4, 0
        >>> r = np.exp(1j * phi) * np.sin(theta)
        >>> t = np.cos(theta)
        >>> optyx_bs = Split(2) @ Split(2) >> Id(1) @ SWAP @ Id(1) \\
        ...            >> Endo(r) @ Endo(t) @ Endo(np.conj(t)) \\
        ...            @ Endo(-np.conj(r)) >> Merge(2) @ Merge(2)
        >>> assert optyx_bs.to_path().prob_with_perceval\\
        ...              (n_photons=1).round(1) == Probabilities[complex](
        ...         [0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
        >>> z_spider = optyx_bs >> Endo(2) @ Id(1) >> optyx_bs
        >>> assert z_spider.to_path().prob_with_perceval\\
        ...               (n_photons=1).round(2)== Probabilities[complex](
        ...         [2.25+0.j, 0.25+0.j, 0.25+0.j, 2.25+0.j], dom=2, cod=2)

        A selection gets the probability of what it selects, not the
        probability of it given that it happened: two photons bunching into
        one arm of a balanced beam splitter is one of two equally likely
        outcomes, so it reads a half, as the Born rule on the amplitudes does.

        >>> bunching = Create(1, 1) >> optyx_bs >> Select(2, 0)
        >>> assert bunching.to_path().prob_with_perceval().round(2) \\
        ...     == bunching.to_path().prob().round(2) \\
        ...     == Probabilities[complex]([0.5+0.j], dom=1, cod=1)
        """
        # pylint: disable=import-outside-toplevel
        import perceval as pcvl

        if not self._umatrix_is_unitary():
            self = self.dilate()

        circ = self._umatrix_to_perceval_circuit()

        proc = pcvl.Processor(simulator)
        proc.set_circuit(circ)

        input_occ = occupation_numbers(n_photons, index(self.dom))
        n_photons_out = (
            n_photons + sum(self.creations) - sum(self.selections))
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        output_occ = occupation_numbers(n_photons_out, index(self.cod))

        states = [pcvl.BasicState(o + self.creations) for o in input_occ]
        selected = [
            pcvl.BasicState(o + self.selections) for o in output_occ]

        born = abs(
            self.scalar
            * self.normalisation ** (n_photons + sum(self.creations))
        ) ** 2

        result = np.zeros((len(states), len(selected)))
        for i, state in enumerate(states):
            proc.with_input(state)
            distribution = proc.probs()["results"]
            result[i] = [born * distribution[o] for o in selected]
        if as_tensor:
            return amplitudes_2_tensor(result, input_occ, output_occ)
        return Probabilities[self.dtype](
            result,
            dom=len(states),
            cod=len(selected),
        )

    def _umatrix_to_perceval_circuit(self) -> pcvl.components.Unitary:
        """The unitary as a single perceval component, simulated as it is
        rather than synthesised into beam splitters and phase shifters."""
        # pylint: disable=import-outside-toplevel
        import perceval as pcvl

        # pylint: disable-next=abstract-class-instantiated  # perceval's
        return pcvl.components.Unitary(pcvl.Matrix(self.array.T.conj()))

    def _umatrix_is_unitary(self) -> bool:
        m = self.umatrix.array
        return np.allclose(np.eye(m.shape[0]), m.dot(m.conj().T))


class Sum(formal_sum.Sum):
    """
    A formal sum of :class:`Matrix` with the same number of modes in and out,
    what :code:`to_path` sends a sum of diagrams to.

    A sum of linear-optical maps with different creations and selections is in
    general not one :class:`Matrix`, since a matrix with :code:`creations` has
    more rows than modes in, so the sum stays formal until a number of photons
    is chosen. On the :code:`n`-photon sector the terms are
    :class:`Amplitudes`, matrices over the occupation numbers of :code:`n`
    photons, and those do add however the terms create and select.

    Parameters:
        terms : The terms of the formal sum.
        dom : The number of modes in, shared by the terms.
        cod : The number of modes out, shared by the terms.
        dtype : The ring the entries live in, that of the terms by default,
            which an empty sum takes from :meth:`Matrix.zero`.

    Example
    -------
    >>> array = np.array([[1, 1], [1, 0]])
    >>> a = Matrix(array, 1, 1, creations=(1, ), selections=(1, ), scalar=2j)
    >>> b = Matrix(array, 1, 1, creations=(1, ), selections=(1, ), scalar=5j)
    >>> assert a.eval(2) + b.eval(2) == (a + b).eval(2)
    >>> (a + b).eval(2)
    Amplitudes([0.+14.j], dom=Nat(1), cod=Nat(1))

    The empty sum is the zero of a homset, see :meth:`Matrix.zero`.

    >>> Matrix.zero(1, 1).eval(2)
    Amplitudes([0.+0.j], dom=Nat(1), cod=Nat(1))
    >>> assert Matrix[float].zero(1, 1).dtype is float
    """
    ob = Nat
    __hash__ = None

    def __init__(self, terms, dom=None, cod=None, dtype=None):
        terms = tuple(terms)
        self.dtype = terms[0].dtype if terms\
            else complex if dtype is None else dtype
        super().__init__(terms, dom, cod)

    @property
    def sum_factory(self) -> Callable[..., Sum]:
        """
        The factory a sum builds sums with, which fixes its ``dtype``, so
        that the ring survives every operation of
        :class:`discopy.sum.Sum` — ``dagger``, ``then``, ``tensor``,
        ``subs`` — including the ones that leave no term to read it from.
        """
        return lambda terms, dom=None, cod=None: type(self)(
            terms, dom, cod, self.dtype)

    def setoid(self):
        """
        A sum is faithfully described by its terms and its boundary, where
        :meth:`discopy.sum.Sum.setoid` reads a singleton sum as its only
        term: a :class:`Matrix` is not an arrow, so it has no setoid. A
        matrix is not hashable either, so neither is a sum: ``__hash__`` is
        ``None`` rather than the :meth:`discopy.cat.Arrow.__hash__` that
        would raise on the terms.

        The ``dtype`` is part of it, as it is of a matrix's own equality,
        since two empty sums over different rings evaluate over those rings.
        """
        return (factory_key(type(self)), self.terms,
                self.dom, self.cod, self.dtype)

    def __repr__(self):
        """
        The representation names the ``dtype``, which a matrix's own carries
        in its factory name, so that a sum reads back equal to itself.
        """
        return f"{self.name[:-1]}, dtype={factory_name(self.dtype)})"

    def to_tree(self) -> dict:
        """ The tree names the ``dtype`` too, as the representation does. """
        return dict(super().to_tree(), dtype=factory_name(self.dtype))

    @classmethod
    def from_tree(cls, tree: dict) -> Sum:
        """ Reads the ``dtype`` back, see :meth:`to_tree`. """
        formal = super().from_tree(tree)
        return cls(formal.terms, formal.dom, formal.cod,
                   resolve_factory(tree["dtype"]))

    @unbiased
    def tensor(self, other: Matrix | Sum) -> Sum:
        """ The tensor of sums distributes over the sum. """
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        return self.sum_factory(
            tuple(f.tensor(g) for f in self.terms for g in other.terms),
            self.dom @ other.dom, self.cod @ other.cod)

    def eval(
        self, n_photons=0, permanent=npperm, as_tensor=False
    ) -> Amplitudes:
        """
        Adds the :class:`Amplitudes` of the terms on one photon sector.

        The terms must send that sector to one sector, i.e. share their
        :attr:`Matrix.net_photons`, or their amplitudes are indexed by
        different occupation numbers: a sum of as many modes out as the
        basis of either sector has elements would add them all the same,
        one mode out being the case where every sector has one element.
        """
        sectors = {term.net_photons for term in self.terms}
        if len(sectors) > 1:
            raise AxiomError(
                "Expected the terms of a sum to add the same number of "
                f"photons, got {sorted(sectors)}.")
        terms = [term.eval(n_photons, permanent, as_tensor)
                 for term in self.terms]
        if terms:
            return sum(terms[1:], terms[0])
        basis = [occupation_numbers(n_photons, index(boundary))
                 for boundary in (self.dom, self.cod)]
        array = np.zeros(tuple(map(len, basis)), dtype=self.dtype)
        return amplitudes_2_tensor(array, *basis) if as_tensor\
            else Amplitudes[self.dtype](array, *map(len, basis))

    def prob(
        self,
        n_photons=0,
        permanent=npperm,
        with_perceval=False,
        as_tensor=False,
    ) -> Probabilities:
        """
        The Born rule of the amplitudes a sum evaluates to, with the
        signature of :meth:`Matrix.prob`. Perceval simulates one unitary,
        which a formal sum of matrices is not, so ``with_perceval`` raises.
        """
        if with_perceval:
            raise NotImplementedError(
                "Perceval simulates one unitary, not a formal sum.")
        amplitudes = self.eval(n_photons, permanent, as_tensor)
        probabilities = np.abs(amplitudes.array) ** 2
        if as_tensor:
            return Tensor(probabilities, amplitudes.dom, amplitudes.cod)
        return Probabilities[amplitudes.dtype](
            probabilities, amplitudes.dom, amplitudes.cod)


class Amplitudes(underlying.Matrix):
    """
    Operator on the Fock space represented as matrix over `occupation_numbers`.

    Example
    -------
    >>> from discopy.optyx.core.zw import Select, Id
    >>> from discopy.optyx.photonic import BS
    >>> BS.get_kraus().to_path().eval(1)
    Amplitudes([0.    +0.70710678j, 0.70710678+0.j    , 0.70710678+0.j    ,
     0.    +0.70710678j], dom=Nat(2), cod=Nat(2))
    >>> assert isinstance(BS.get_kraus().to_path().eval(2), Amplitudes)
    >>> assert np.allclose((BS.get_kraus() >> Select(1) @ \\
    ... Id(1)).to_path().eval(2).array,\\
    ... Amplitudes([0.+0.70710678j, -0.+0.j    , 0.+0.70710678j], \\
    ... dom=3, cod=1).array)
    """

    dtype = complex

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)


class Probabilities(underlying.Matrix):
    """
    Stochastic matrix of probabilities over `occupation_numbers`.

    Example
    -------
    >>> from discopy.optyx.core.zw import Create
    >>> from discopy.optyx.photonic import BS
    >>> probabilities = BS.get_kraus().to_path().prob(1).round(1)
    >>> probabilities  # doctest: +NORMALIZE_WHITESPACE
    Probabilities[complex]([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j],
                           dom=Nat(2), cod=Nat(2))
    >>> hom = (Create(1, 1) >> BS.get_kraus()).to_path()
    >>> hom.prob().round(1)  # doctest: +NORMALIZE_WHITESPACE
    Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j],
                           dom=Nat(1), cod=Nat(3))
    """

    dtype = float

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def normalise(self) -> Probabilities:
        return self.__class__(
            array=self.array / self.array.sum(axis=1)[:, None],
            dom=self.dom,
            cod=self.cod,
        )


Matrix.sum_factory = Sum
