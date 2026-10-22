# -*- coding: utf-8 -*-

"""
Gates in a :class:`discopy.quantum.circuit.Circuit`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    SelfConjugate
    AntiConjugate
    Discard
    MixedState
    Measure
    Encode
    QuantumGate
    ClassicalGate
    Copy
    Match
    Digits
    Bits
    Ket
    Bra
    Controlled
    Parametrized
    Rotation
    Rx
    Ry
    Rz
    CU1
    CRz
    CRx
    Scalar
    MixedScalar
    Sqrt

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        sqrt
        scalar
"""
import copy
from collections.abc import Sized
from math import e, pi

from discopy import messages
from discopy.feedback import str_delayed
from discopy.cat import rsubs
from discopy.matrix import get_backend
from discopy.quantum.circuit import (
    Circuit, Digit, Ty, bit, qubit, Box, Sum, Id)
from discopy.tensor import backend
from discopy.utils import factory_name, assert_isinstance, from_tree


def format_number(data):
    """ Tries to format a number. """
    try:
        return f'{data:.3g}'
    except TypeError:
        return data


class SelfConjugate(Box):
    """ A self-conjugate box, i.e. where the transpose is the dagger. """
    def conjugate(self):
        return self

    def rotate(self, left=False):
        del left
        return self.dagger()


class AntiConjugate(Box):
    """ An anti-conjugate box, i.e. where the conjugate is the dagger. """
    def conjugate(self):
        return self.dagger()

    def rotate(self, left=False):
        del left
        return self


class Discard(SelfConjugate):
    """
    Discard n qubits. If :code:`dom == bit` then marginal distribution.

    A gate is causal, i.e. discarding all its outputs is the same as
    discarding all its inputs.

    >>> from discopy.monoidal import Equation
    >>> Equation(CX >> Discard(2), Discard(2)).draw(
    ...     wire_labels=False, doctest="docs/_static/quantum/discard.svg")

    .. image:: /_static/quantum/discard.svg
        :align: center

    >>> assert (CX >> Discard(2)).eval() == Discard(2).eval()

    Post-selection is not causal: discarding the output of a bra is not the
    discard of its input.

    >>> assert Bra(0).eval(mixed=True) != Discard(1).eval()
    """
    draw_as_discards = True

    def __init__(self, dom=1):
        if isinstance(dom, int):
            dom = qubit ** dom
        super().__init__(
            f"Discard({dom})", dom, qubit ** 0, is_mixed=True)
        self.n_qubits = len(dom)

    def __getnewargs__(self):
        """
        The argument of the constructor, which a delay reads: a discard is
        delayed by delaying the type it discards.

        >>> assert Discard(2).delay() == Discard(qubit.delay() ** 2)
        """
        return (self.dom, )

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(from_tree(tree['dom']))

    def dagger(self):
        return MixedState(self.dom)

    def _decompose(self):
        return Id().tensor(*map(Discard, self.dom))


class MixedState(SelfConjugate):
    """
    Maximally-mixed state on n qubits.
    If :code:`cod == bit` then uniform distribution.
    """
    draw_as_discards = True

    def __init__(self, cod=1):
        if isinstance(cod, int):
            cod = qubit ** cod
        super().__init__(
            f"MixedState({cod})", qubit ** 0, cod, is_mixed=True)
        self.drawing_name = "MixedState"
        if cod == bit:
            self.drawing_name = ""
            self.draw_as_spider, self.color = True, "black"

    def __getnewargs__(self):
        return (self.cod, )

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(from_tree(tree['cod']))

    def dagger(self):
        return Discard(self.cod)

    def _decompose(self):
        return Id().tensor(*map(MixedState, self.cod))


class Measure(SelfConjugate):
    """
    Measure n qubits into n bits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to measure.
    destructive : bool, optional
        Whether to do a non-destructive measurement instead.
    override_bits : bool, optional
        Whether to override input bits, this is the standard behaviour of tket.
    time_step : int, optional
        The number of times the measurement has been delayed.
    """
    draw_as_measures = True

    def __init__(self, n_qubits=1, destructive=True, override_bits=False,
                 time_step=0):
        dom, cod = qubit ** n_qubits, bit ** n_qubits
        name = f"Measure({'' if n_qubits == 1 else n_qubits})"
        if not destructive:
            cod = qubit ** n_qubits @ cod
            name = name\
                .replace("()", "(1)").replace(')', ", destructive=False)")
        if override_bits:
            dom = dom @ bit ** n_qubits
            name = name\
                .replace("()", "(1)").replace(')', ", override_bits=True)")
        super().__init__(name, dom.delay(time_step), cod.delay(time_step),
                         is_mixed=True, time_step=time_step)
        self.destructive, self.override_bits = destructive, override_bits
        self.n_qubits = n_qubits
        self.draw_as_measures = True

    def delay(self, n_steps=1):
        """
        The same measurement, one more time step on.

        >>> assert Measure(2).delay().dom == qubit.delay() ** 2
        """
        return Measure(self.n_qubits, self.destructive, self.override_bits,
                       self.time_step + n_steps)

    def to_tree(self) -> dict:
        return dict(n_qubits=self.n_qubits, destructive=self.destructive,
                    override_bits=self.override_bits, **super().to_tree())

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree['n_qubits'], tree['destructive'],
                   tree['override_bits'], tree.get('time_step', 0))

    def dagger(self):
        return Encode(self.n_qubits,
                      constructive=self.destructive,
                      reset_bits=self.override_bits,
                      time_step=self.time_step)

    def _decompose(self):
        return Id().tensor(*[
            Measure(destructive=self.destructive,
                    override_bits=self.override_bits,
                    time_step=self.time_step)] * self.n_qubits)


class Encode(SelfConjugate):
    """
    Controlled preparation, i.e. encode n bits into n qubits.

    Parameters
    ----------
    n_bits : int
        Number of bits to encode.
    constructive : bool, optional
        Whether to do a classically-controlled correction instead.
    reset_bits : bool, optional
        Whether to reset the bits to the uniform distribution.
    time_step : int, optional
        The number of times the encoding has been delayed.
    """
    draw_as_measures = True

    def __init__(self, n_bits=1, constructive=True, reset_bits=False,
                 time_step=0):
        dom, cod = bit ** n_bits, qubit ** n_bits
        if not constructive:
            dom = qubit ** n_bits @ dom
        if reset_bits:
            cod = cod @ bit ** n_bits
        name = Measure(n_bits, constructive, reset_bits).name\
            .replace("Measure", "Encode")\
            .replace("destructive", "constructive")\
            .replace("override_bits", "reset_bits")
        super().__init__(name, dom.delay(time_step), cod.delay(time_step),
                         is_mixed=True, time_step=time_step)
        self.constructive, self.reset_bits = constructive, reset_bits
        self.n_bits = n_bits

    def delay(self, n_steps=1):
        return Encode(self.n_bits, self.constructive, self.reset_bits,
                      self.time_step + n_steps)

    def to_tree(self) -> dict:
        return dict(n_bits=self.n_bits, constructive=self.constructive,
                    reset_bits=self.reset_bits, **super().to_tree())

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree['n_bits'], tree['constructive'], tree['reset_bits'],
                   tree.get('time_step', 0))

    def dagger(self):
        return Measure(self.n_bits,
                       destructive=self.constructive,
                       override_bits=self.reset_bits,
                       time_step=self.time_step)

    def _decompose(self):
        return Id().tensor(*[
            Encode(constructive=self.constructive,
                   reset_bits=self.reset_bits,
                   time_step=self.time_step)] * self.n_bits)


class QuantumGate(Box):
    """
    Quantum gates, i.e. unitaries on n qubits.

    Its data is flattened into a list of complex numbers, except for the
    entries with free symbols.

    >>> from sympy.abc import phi  # doctest: +EXTRA
    >>> import numpy as np
    >>> assert QuantumGate('U', qubit, qubit, np.eye(2)).data == [1, 0, 0, 1]
    >>> assert QuantumGate('U', qubit, qubit, [phi, 0, 0, 1]).free_symbols\\
    ...     == {phi}
    """
    is_mixed = False
    is_classical = False

    def __init__(self, name: str, dom: Ty, cod: Ty, data=None, **params):
        if isinstance(data, Sized):
            with backend('numpy') as np:
                data = [value if getattr(value, "free_symbols", None)
                        else complex(value) for value in np.ravel(data)]
        super().__init__(name, dom, cod, data, **params)

    def __setstate__(self, state):
        if "_array" in state and not state["_array"] is None:
            state["data"] = state['_array'].flatten().tolist()
        if "_name" in state:
            if state["_name"] in GATES and hasattr(
                    GATES[state["_name"]], "data"):
                state["data"] = copy.deepcopy(GATES[state["_name"]].data)
                state["_z"] = GATES[state["_name"]].z
        super().__setstate__(state)

    def setoid(self):
        """ Avoid checking for equality of matrices when comparing gates. """
        return (repr(self), )


class ClassicalGate(SelfConjugate):
    """
    Classical gates, i.e. from digits to digits.

    >>> from sympy import symbols  # doctest: +EXTRA
    >>> array = symbols("a b c d")
    >>> f = ClassicalGate('f', bit, bit, array)
    >>> f.data
    (a, b, c, d)
    >>> f.lambdify(*array)(1, 2, 3, 4).data
    (1, 2, 3, 4)
    """
    is_mixed = False
    is_classical = True


class Copy(ClassicalGate):
    """ Takes a bit, returns two copies of it. """
    def __init__(self, time_step=0):
        dom = bit.delay(time_step)
        super().__init__("Copy", dom, dom ** 2, [1, 0, 0, 0, 0, 0, 0, 1],
                         time_step=time_step)
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    def delay(self, n_steps=1):
        """
        The same copy, one more time step on.

        >>> assert Copy().delay().dagger() == Match().delay()
        """
        return type(self)(self.time_step + n_steps)

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree.get('time_step', 0))

    def dagger(self):
        return Match(self.time_step)


class Match(ClassicalGate):
    """ Takes two bits in, returns them if they are equal. """
    def __init__(self, time_step=0):
        cod = bit.delay(time_step)
        super().__init__("Match", cod ** 2, cod, [1, 0, 0, 0, 0, 0, 0, 1],
                         time_step=time_step)
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    delay = Copy.delay
    from_tree = classmethod(Copy.from_tree.__func__)

    def dagger(self):
        return Copy(self.time_step)


class Digits(ClassicalGate):
    """
    Classical state for a given string of digits of a given dimension.

    Examples
    --------
    >>> from discopy.tensor import Dim, Tensor
    >>> assert Digits(2, dim=4).eval()\\
    ...     == Tensor[complex](dom=Dim(1), cod=Dim(4), array=[0, 0, 1, 0])
    """
    draw_as_brakets = True

    def __init__(self, *digits, dim=None, is_dagger=False, time_step=0):
        if not isinstance(dim, int):
            raise TypeError(int, dim)
        if not all(digit in range(dim) for digit in digits):
            raise ValueError(
                f"Digits of dimension {dim} are integers from 0 to {dim - 1}.")
        digits = tuple(map(int, digits))
        self._digits, self._dim = digits, dim
        str_digits = ', '.join(map(str, digits))
        name = f"Digits({str_digits}, dim={dim})" if dim != 2 \
            else f"Bits({str_digits})"
        dom, cod = Ty(), Ty(Digit(dim)).delay(time_step) ** len(digits)
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        super().__init__(
            name, dom, cod, is_dagger=is_dagger, time_step=time_step)

    def __repr__(self):
        """
        The constructor call of a classical state, through :func:`Bits` for
        bits, so that ``eval(repr(x)) == x``.

        >>> from discopy.quantum.circuit import Circuit
        >>> for state in (Bits(1, 0).dagger(), Digits(2, dim=4).delay()):
        ...     assert eval(repr(state), Circuit.environment()) == state
        """
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        digits = ', '.join(map(repr, self.digits))
        factory, dim = ("Bits", "") if self.dim == 2\
            else ("Digits", f", dim={self.dim}")
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        module = type(self).__module__.removeprefix("discopy.")
        return f"{module}.{factory}({digits}{dim}{time_step})"

    def delay(self, n_steps=1):
        return Digits(*self.digits, dim=self.dim, is_dagger=self.is_dagger,
                      time_step=self.time_step + n_steps)

    def to_tree(self) -> dict:
        return dict(digits=self.digits, dim=self.dim, **super().to_tree())

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(*tree['digits'], dim=tree['dim'],
                   is_dagger=tree.get('is_dagger', False),
                   time_step=tree.get('time_step', 0))

    @property
    def dim(self):
        """
        The dimension of the information units.

        >>> assert Bits(1, 0).dim == 2
        """
        return self._dim

    @property
    def digits(self):
        """ The digits of a classical state. """
        return list(self._digits)

    bitstring = digits

    @property
    def array(self):
        with backend('numpy') as np:
            array = np.zeros(len(self._digits) * (self._dim, ))
            array[self._digits] = 1
            return array

    def dagger(self):
        return Digits(*self.digits, dim=self.dim, is_dagger=not self.is_dagger,
                      time_step=self.time_step)

    def to_drawing(self):
        result = QuantumGate.to_drawing(self)
        result.draw_as_brakets, result._digits = True, self._digits
        return result


def Bits(*bitstring, is_dagger=False, time_step=0):
    """ Classical state for a given bitstring, i.e. digits of dimension 2. """
    return Digits(*bitstring, dim=2, is_dagger=is_dagger, time_step=time_step)


class Ket(SelfConjugate, QuantumGate):
    """
    Implements qubit preparation for a given bitstring.

    >>> from discopy.tensor import Dim, Tensor
    >>> assert Ket(1, 0).cod == qubit ** 2
    >>> assert Ket(1, 0).eval()\\
    ...     == Tensor[complex](dom=Dim(1), cod=Dim(2, 2), array=[0, 0, 1, 0])
    """
    to_drawing = Digits.to_drawing
    array = Digits.array

    def __init__(self, *bitstring, time_step=0):
        if not all([bit in [0, 1] for bit in bitstring]):
            raise Exception('Bitstring can only contain integers 0 or 1.')
        bitstring = tuple(map(int, bitstring))
        dom, cod = qubit ** 0, qubit.delay(time_step) ** len(bitstring)
        name = f"Ket({', '.join(map(str, bitstring))})"
        super().__init__(name, dom, cod, time_step=time_step)
        self._digits, self._dim, self.draw_as_brakets = bitstring, 2, True

    def delay(self, n_steps=1):
        """
        The preparation of the same bitstring, one more time step on.

        >>> assert Ket(1, 0).delay().dagger() == Bra(1, 0).delay()
        """
        return type(self)(*self.bitstring, time_step=self.time_step + n_steps)

    def to_tree(self) -> dict:
        return dict(bitstring=self.bitstring, **SelfConjugate.to_tree(self))

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(*tree['bitstring'], time_step=tree.get('time_step', 0))

    @property
    def bitstring(self):
        """ The bitstring of a Ket. """
        return list(self._digits)

    def dagger(self):
        return Bra(*self.bitstring, time_step=self.time_step)

    def _decompose(self):
        return Id().tensor(*[
            Ket(b, time_step=self.time_step) for b in self.bitstring])


class Bra(SelfConjugate, QuantumGate):
    """
    Implements qubit post-selection for a given bitstring.

    >>> from discopy.tensor import Dim, Tensor
    >>> assert Bra(1, 0).dom == qubit ** 2
    >>> assert Bra(1, 0).eval()\\
    ...     == Tensor[complex](dom=Dim(2, 2), cod=Dim(1), array=[0, 0, 1, 0])
    """
    to_drawing = Digits.to_drawing
    array = Digits.array

    def __init__(self, *bitstring, time_step=0):
        if not all([bit in [0, 1] for bit in bitstring]):
            raise Exception('Bitstring can only contain integers 0 or 1.')
        bitstring = tuple(map(int, bitstring))
        name = f"Bra({', '.join(map(str, bitstring))})"
        dom, cod = qubit.delay(time_step) ** len(bitstring), qubit ** 0
        super().__init__(name, dom, cod, time_step=time_step)
        self._digits, self._dim, self.draw_as_brakets = bitstring, 2, True

    delay, to_tree = Ket.delay, Ket.to_tree
    from_tree = classmethod(Ket.from_tree.__func__)

    @property
    def bitstring(self):
        """ The bitstring of a Bra. """
        return list(self._digits)

    def dagger(self):
        return Ket(*self.bitstring, time_step=self.time_step)

    def _decompose(self):
        return Id().tensor(*[
            Bra(b, time_step=self.time_step) for b in self.bitstring])


class Controlled(QuantumGate):
    """
    Abstract class for controled quantum gates.

    Parameters
    ----------
    controlled : QuantumGate
        Gate to control, e.g. :code:`CX = Controlled(X)`.
    distance : int, optional
        Number of qubits from the control to the target, default is :code:`0`.
        If negative, the control is on the right of the target.

    >>> circuit = (Controlled(CX.l, distance=3)
    ...            >> Controlled(Controlled(CZ.l, distance=2), distance=-1))
    >>> circuit.draw(
    ...     wire_labels=False,
    ...     doctest="docs/_static/quantum/long-controlled.svg")

    .. image:: /_static/quantum/long-controlled.svg
        :align: center
    """
    draw_as_controlled = True

    def __init__(self, controlled, distance=1):
        assert_isinstance(controlled, QuantumGate)
        if not distance:
            raise ValueError(messages.ZERO_DISTANCE_CONTROLLED)
        self.controlled, self.distance = controlled, distance
        n_qubits = len(controlled.dom) + abs(distance)
        name = f'C{controlled.reset()}'
        time_step = controlled.time_step
        dom = cod = qubit.delay(time_step) ** n_qubits
        QuantumGate.__init__(self, name, dom, cod, data=controlled.data,
                             time_step=time_step)

    def __getnewargs__(self):
        """
        The arguments of the constructor, which a delay reads: a controlled
        gate is at the time step of the gate it controls.

        >>> assert CX.delay() == Controlled(X.delay())
        """
        return self.controlled, self.distance

    def with_distance(self, distance) -> "Controlled":
        """
        The same controlled gate with the control at a given ``distance``.

        Example
        -------
        >>> assert CRz(0.25, distance=-1).with_distance(1) == CRz(0.25)
        """
        return type(self)(self.controlled, distance=distance)

    def dagger(self):
        return Controlled(self.controlled.dagger(), distance=self.distance)

    def conjugate(self):
        controlled_conj = self.controlled.conjugate()
        return Controlled(controlled_conj, distance=-self.distance)

    def rotate(self, left=False):
        """
        The transpose of a controlled gate controls the transpose of its
        gate, with the order of the qubits reversed.

        >>> assert CRz(.3).l.eval().is_close(CRz(.3).eval().l)
        """
        return Controlled(
            self.controlled.rotate(left), distance=-self.distance)

    def lambdify(self, *symbols, **kwargs):
        c_fn = self.controlled.lambdify(*symbols)
        return lambda *xs: type(self)(c_fn(*xs), distance=self.distance)

    def subs(self, *args):
        controlled = self.controlled.subs(*args)
        return type(self)(controlled, distance=self.distance)

    def __repr__(self):
        """
        The constructor call of a controlled gate with its module, the same
        for a subclass such as :class:`CRz`, which is equal to the gate it
        abbreviates and so hashes the same.

        >>> from discopy.quantum.circuit import Circuit
        >>> assert eval(repr(CX), Circuit.environment()) == CX
        >>> assert repr(CRz(.5)) == repr(Controlled(Rz(.5)))
        """
        return factory_name(Controlled)\
            + f'({self.controlled!r}, distance={self.distance!r})'

    def __str__(self):
        return self.name + str_delayed(self.time_step) if self.distance == 1\
            else f'Controlled({self.controlled}, distance={self.distance})'

    def __eq__(self, other):
        return not isinstance(other, Box) and super().__eq__(other)\
            or isinstance(other, Controlled)\
            and self.distance == other.distance\
            and self.controlled == other.controlled

    def to_tree(self) -> dict:
        return dict(controlled=self.controlled.to_tree(),
                    distance=self.distance, **super().to_tree())

    @classmethod
    def from_tree(cls, tree: dict):
        """
        Reads back through :class:`Controlled` rather than through ``cls``: a
        subclass such as :class:`CRz` takes the phase of the rotation it
        controls where :class:`Controlled` takes the gate itself, and
        :meth:`__eq__` compares the gate and the distance rather than the
        class, so the two are equal.

        >>> assert from_tree(CRz(.25).to_tree()) == CRz(.25)
        """
        return Controlled(
            from_tree(tree['controlled']), distance=tree['distance'])

    @property
    def phase(self):
        return self.controlled.phase

    __hash__ = QuantumGate.__hash__

    def _decompose_grad(self):
        if self.time_step:
            return Controlled._decompose_grad(self.reset()).delay(
                self.time_step)
        controlled, distance = self.controlled, self.distance
        if isinstance(controlled, (Rx, Rz)):
            phase = self.phase
            decomp = Controlled(X, distance=distance)\
                >> qubit ** distance @ Rz(-phase / 2) @ qubit ** -distance\
                >> Controlled(X, distance=distance)\
                >> qubit ** distance @ Rz(phase / 2) @ qubit ** -distance
            if isinstance(controlled, Rx):
                decomp <<= qubit ** distance @ H @ qubit ** -distance
                decomp >>= qubit ** distance @ H @ qubit ** -distance
            return decomp
        return self

    def _decompose(self):
        controlled, distance = self.controlled, self.distance
        n_qubits = len(self.dom)

        if distance == 1:
            return self

        skipped_qbs = n_qubits - (1 + len(controlled.dom))

        if distance > 0:
            pattern = [0,
                       *range(skipped_qbs + 1, n_qubits),
                       *range(1, skipped_qbs + 1)]
        else:
            pattern = [n_qubits - 1, *range(n_qubits - 1)]

        perm = Circuit.permutation(pattern, self.dom)
        diagram = (perm
                   >> self.with_distance(1) @ self.dom[:skipped_qbs]
                   >> perm[::-1])

        return diagram

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        decomp = self._decompose_grad()
        if decomp == self:
            raise NotImplementedError()
        return decomp.grad(var, **params)

    @property
    def array(self):
        """
        The array of a controlled gate, built from the array of the gate it
        controls as that gate is, i.e. transposed when it is rotated and
        conjugate transposed when it is a dagger.

        >>> import numpy as np
        >>> CS = Controlled(S)
        >>> assert np.allclose(CS.dagger().array.reshape(4, 4),
        ...                    CS.array.reshape(4, 4).T.conj())
        """
        controlled, distance = self.controlled, self.distance
        n_qubits = len(self.dom)
        if distance != 1:
            return self._decompose().eval().array.reshape(*[2] * 2 * n_qubits)
        d = 1 << n_qubits - 1
        with backend() as np:
            target = np.array(controlled.array)
            target = (target.transpose() if controlled.z else target)\
                .reshape(d, d)
            target = np.conj(target).T if controlled.is_dagger else target
            array = np.kron(np.array([[1, 0], [0, 0]]), np.eye(d))\
                + np.kron(np.array([[0, 0], [0, 1]]), target)
        return array.reshape(*[2] * 2 * n_qubits)

    def to_drawing(self):
        result = super().to_drawing()
        result.distance, result.controlled = self.distance, self.controlled
        return result


class Parametrized(Box):
    """
    Abstract class for parametrized boxes in a quantum circuit.

    Parameters
    ----------
    name : str
        Name of the parametrized class, e.g. :code:`"CRz"`.
    dom, cod : discopy.quantum.circuit.Ty
        Domain and codomain.
    data : any
        Data of the box, potentially with free symbols.

    Example
    -------
    >>> from sympy.abc import phi  # doctest: +EXTRA
    >>> from sympy import pi, exp, I
    >>> assert Rz(phi).array[0,0] == exp(-1.0 * I * pi * phi)
    >>> c = Rz(phi) >> Rz(-phi)
    >>> assert c.lambdify(phi)(.25) == Rz(.25) >> Rz(-.25)
    """
    def __init__(self, name, dom, cod, data=None, **params):
        self.drawing_name = f'{name}({data})'
        Box.__init__(self, name, dom, cod, data=data, **params)

    @property
    def modules(self):
        if self.free_symbols:
            import sympy
            return sympy
        else:
            return get_backend()

    def subs(self, *args):
        data = rsubs(self.data, *args)
        return type(self)(data)

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify
        with backend() as np:
            data = lambdify(symbols, self.data, dict(kwargs, modules=np))
        return lambda *xs: type(self)(data(*xs))

    def __str__(self):
        if isinstance(self, Controlled):
            # Ensure `Controlled(Rx(0.5))` and `CRx(0.5)` are printed the same.
            return Controlled.__str__(self)
        return f'{self.name}({format_number(self.data)})'\
            + str_delayed(self.time_step)

    def __repr__(self):
        return factory_name(type(self)) + f"({format_number(self.data)})"


class Rotation(Parametrized, QuantumGate):
    """ Abstract class for rotation gates. """
    n_qubits = 1

    def __init__(self, phase, z=0, time_step=0):
        name, n_qubits = type(self).__name__, type(self).n_qubits
        dom = cod = qubit.delay(time_step) ** n_qubits
        QuantumGate.__init__(self, name, dom, cod, z=z, time_step=time_step)
        Parametrized.__init__(self, name, dom, cod, is_mixed=False, data=phase,
                              time_step=time_step)

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree['data'], tree.get('z', 0), tree.get('time_step', 0))

    @property
    def phase(self):
        """ The phase of a rotation gate. """
        return self.data

    def delay(self, n_steps=1):
        """
        The rotation by the same phase, one more time step on.

        >>> assert Rz(.5).delay().dom == qubit.delay()
        """
        return type(self)(self.phase, self.z, self.time_step + n_steps)

    def dagger(self):
        return type(self)(-self.phase, time_step=self.time_step)

    def rotate(self, left=False):
        del left
        return type(self)(
            self.phase, z=int(not self.z), time_step=self.time_step)

    def subs(self, *args):
        return super().subs(*args).delay(self.time_step)

    def lambdify(self, *symbols, **kwargs):
        function = super().lambdify(*symbols, **kwargs)
        return lambda *xs: function(*xs).delay(self.time_step)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient

        with backend() as np:
            if params.get('mixed', True):
                if len(self.dom) != 1:
                    raise NotImplementedError
                s = scalar(np.pi * gradient, is_mixed=True)
                t1 = type(self)(self.phase + .25).delay(self.time_step)
                t2 = type(self)(self.phase - .25).delay(self.time_step)
                return s @ (t1 + scalar(-1, is_mixed=True) @ t2)

            return scalar(np.pi * gradient)\
                @ type(self)(self.phase + .5).delay(self.time_step)


class Rx(AntiConjugate, Rotation):
    """ X rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase, dtype=complex)
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)
            return np.stack((cos, -1j * sin, -1j * sin, cos)).reshape(2, 2)


class Ry(SelfConjugate, Rotation):
    """ Y rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase)
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)
            return np.stack((cos, sin, -sin, cos)).reshape(2, 2)


class Rz(AntiConjugate, Rotation):
    """ Z rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase)
            e1 = self.modules.exp(-1j * half_theta)
            e2 = self.modules.exp(1j * half_theta)
            z = np.array(0)
            return np.stack((e1, z, z, e2)).reshape(2, 2)


class U1(AntiConjugate, Rotation):
    """ Z rotation, differ from :class:`Rz` by a global phase. """
    def grad(self, var, **params):
        """
        The derivative of a U1 gate: the one of :class:`Rz` under mixed
        evaluation, where the global phase cancels, and under pure evaluation
        the difference of the gate shifted by a quarter turn either way, as
        the gate is no multiple of its shift by half a turn.

        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> import numpy as np
        >>> grad = U1(phi).grad(phi, mixed=False).subs(phi, .3).eval()
        >>> assert np.allclose(grad.array, [[0, 0], [0, 2j * np.pi * (
        ...     U1(.3).array[1, 1])]])
        """
        if params.get('mixed', True) or var not in self.free_symbols:
            return super().grad(var, **params)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        plus, minus = (
            U1(self.phase + shift).delay(self.time_step)
            for shift in (.25, -.25))
        with backend() as np:
            return scalar(np.pi * gradient) @ (plus + scalar(-1) @ minus)

    @property
    def array(self):
        with backend() as np:
            theta = np.array(2 * self.modules.pi * self.phase)
            return np.stack(
                (1, 0, 0, self.modules.exp(1j * theta))).reshape(2, 2)


class ControlledRotation(Controlled, Rotation):
    """ Controlled rotation gate. """
    def __init__(self, phase, distance=1, time_step=0):
        Controlled.__init__(
            self, self.controlled(phase, time_step=time_step), distance)

    def delay(self, n_steps=1):
        return type(self)(
            self.phase, self.distance, self.time_step + n_steps)

    def with_distance(self, distance) -> "ControlledRotation":
        return type(self)(self.phase, distance, self.time_step)

    def subs(self, *args):
        """
        Substitute in the phase, at the same distance and time step.

        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> assert CRz(phi, distance=2).subs(phi, .5) == CRz(.5, distance=2)
        """
        return type(self)(
            self.controlled.subs(*args).phase, self.distance, self.time_step)

    def lambdify(self, *symbols, **kwargs):
        function = self.controlled.lambdify(*symbols, **kwargs)
        return lambda *xs: type(self)(
            function(*xs).phase, self.distance, self.time_step)


class CU1(ControlledRotation):
    """ Controlled U1 rotations. """
    controlled = U1


class CRz(ControlledRotation):
    """ Controlled Z rotations. """
    controlled = Rz


class CRx(ControlledRotation):
    """ Controlled X rotations. """
    controlled = Rx


class Scalar(Parametrized):
    """ Scalar, i.e. quantum gate with empty domain and codomain. """
    def __init__(self, data, name=None, is_mixed=False, time_step=0):
        self.drawing_name = format_number(data)
        name = "scalar" if name is None else name
        dom, cod = qubit ** 0, qubit ** 0
        super().__init__(name, dom, cod, is_mixed=is_mixed, data=data, z=None,
                         time_step=time_step)

    def __setstate__(self, state):
        state["_z"] = None
        super().__setstate__(state)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + (
            ', is_mixed=True' if self.is_mixed else '') + time_step + ')'

    def setoid(self):
        """
        A scalar and its delay differ only by their time step, as the delay
        of an empty boundary is itself.

        >>> assert scalar(2).delay() != scalar(2) == scalar(2).d.delay(-1)
        """
        return super().setoid() + (self.time_step, )

    def delay(self, n_steps=1):
        return type(self)(
            self.data, self.name, self.is_mixed, self.time_step + n_steps)

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data)

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.data.diff(var), is_mixed=self.is_mixed,
                      time_step=self.time_step)

    def dagger(self):
        return Scalar(self.data.conjugate(), self.name, self.is_mixed,
                      self.time_step)

    def subs(self, *args):
        """
        Substitute in the data of a scalar, with the same name, mixed when it
        is.

        >>> from sympy.abc import phi  # doctest: +EXTRA
        >>> assert scalar(2 * phi, is_mixed=True).subs(phi, .5).is_mixed
        """
        return type(self)(rsubs(self.data, *args), self.name, self.is_mixed,
                          self.time_step)

    def lambdify(self, *symbols, **kwargs):
        function = super().lambdify(*symbols, **kwargs)
        return lambda *xs: type(self)(
            function(*xs).data, self.name, self.is_mixed, self.time_step)

    def to_tree(self) -> dict:
        return dict(is_mixed=self.is_mixed, **super().to_tree())

    @classmethod
    def from_tree(cls, tree: dict):
        """
        Passes its arguments positionally, as :meth:`dagger` builds one:
        :meth:`tensor.Box.__new__` reads ``name`` as its own first argument,
        where a scalar's first argument is its ``data``.

        >>> assert from_tree(Scalar(.5).to_tree()) == Scalar(.5)
        """
        return cls(tree['data'], tree['name'], tree['is_mixed'],
                   tree.get('time_step', 0))


class MixedScalar(Scalar):
    """ Mixed scalar, i.e. where the Born rule has already been applied. """
    def __init__(self, data, time_step=0):
        super().__init__(data, is_mixed=True, time_step=time_step)

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree['data'], tree.get('time_step', 0))

    def delay(self, n_steps=1):
        """
        The same mixed scalar, one more time step on.

        >>> assert isinstance(MixedScalar(.5).delay(), MixedScalar)
        """
        return MixedScalar(self.data, self.time_step + n_steps)

    def subs(self, *args):
        return Parametrized.subs(self, *args).delay(self.time_step)

    def lambdify(self, *symbols, **kwargs):
        function = Parametrized.lambdify(self, *symbols, **kwargs)
        return lambda *xs: function(*xs).delay(self.time_step)


class Sqrt(Scalar):
    """
    Principal square root, i.e. the one with non-negative real part. It is
    self-adjoint when that root is real, and conjugated by its dagger
    otherwise, e.g. for the square root of a negative number.
    """
    def __init__(self, data, is_dagger=False, time_step=0):
        super().__init__(data, name="sqrt", time_step=time_step)
        if is_dagger and self.is_self_adjoint:
            raise ValueError(messages.REAL_SQRT_IS_SELF_ADJOINT)
        self.is_dagger = is_dagger
        self.drawing_name = f"sqrt({format_number(data)})"

    @classmethod
    def from_tree(cls, tree: dict):
        """
        Reads the dagger flag a box's tree keeps, so that the square root of
        a negative number comes back conjugated when it was.

        >>> root = Sqrt(-1).dagger()
        >>> assert from_tree(root.to_tree()) == root
        """
        return cls(tree['data'], tree.get('is_dagger', False),
                   tree.get('time_step', 0))

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.is_dagger is None:
            self.is_dagger = False

    @property
    def is_self_adjoint(self):
        """ Whether the root is real, i.e. the dagger is the identity. """
        return not self.free_symbols and not complex(self.data ** .5).imag

    @property
    def array(self):
        with backend() as np:
            root = np.array(self.data ** .5)
            return root.conjugate() if self.is_dagger else root

    def dagger(self):
        return self if self.is_self_adjoint\
            else Sqrt(self.data, not self.is_dagger, self.time_step)

    def delay(self, n_steps=1):
        return Sqrt(self.data, self.is_dagger, self.time_step + n_steps)

    def subs(self, *args):
        result = Parametrized.subs(self, *args).delay(self.time_step)
        return result.dagger() if self.is_dagger else result

    def lambdify(self, *symbols, **kwargs):
        function = Parametrized.lambdify(self, *symbols, **kwargs)
        return lambda *xs: function(*xs).delay(self.time_step).dagger()\
            if self.is_dagger else function(*xs).delay(self.time_step)

    def setoid(self):
        """ A square root and its dagger differ only by ``is_dagger``. """
        return super().setoid() + (self.is_dagger, )

    def __repr__(self):
        return super().__repr__() + ("[::-1]" if self.is_dagger else "")


def sqrt(expr):
    """ Returns a 0-qubit quantum gate that scales by a square root. """
    return Sqrt(expr)


def scalar(expr, is_mixed=False):
    """ Returns a 0-qubit quantum gate that scales by a complex number. """
    return Scalar(expr, is_mixed=is_mixed)


SWAP = QuantumGate(
    'SWAP', qubit ** 2, qubit ** 2, draw_as_wires=True, is_dagger=None,
    z=None, data=[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
H = QuantumGate(
    'H', qubit, qubit,
    data=[2 ** -0.5 * x for x in [1, 1, 1, -1]], is_dagger=None, z=None)
S = QuantumGate('S', qubit, qubit, [1, 0, 0, 1j])
T = QuantumGate('T', qubit, qubit, [1, 0, 0, e ** (1j * pi / 4)])
X = QuantumGate('X', qubit, qubit, [0, 1, 1, 0], is_dagger=None, z=None)
Y = QuantumGate('Y', qubit, qubit, [0, 1j, -1j, 0], is_dagger=None)
Z = QuantumGate('Z', qubit, qubit, [1, 0, 0, -1], is_dagger=None, z=None)
CX = Controlled(X)
CY = Controlled(Y)
CZ = Controlled(Z)
CCX = Controlled(CX)
CCZ = Controlled(CZ)

GATES = {
    'SWAP': SWAP,
    'H': H,
    'S': S,
    'T': T,
    'X': X,
    'Y': Y,
    'Z': Z,
    'CZ': CZ,
    'CY': CY,
    'CX': CX,
    'CCX': CCX,
    'CCZ': CCZ,
    'Rx': Rx,
    'Ry': Ry,
    'Rz': Rz,
    'U1': U1,
    'CRx': CRx,
    'CRz': CRz,
    'CU1': CU1,
}


for attr, gate in GATES.items():
    def closure(attr=attr, gate=gate):
        """ Easiest way around the Python late binding gotcha. """
        if isinstance(gate, Controlled)\
                and isinstance(gate.controlled, Controlled):
            def method(self, i: int, j: int, k: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit indices.

                Parameters:
                    i : First control index.
                    j : Second control index.
                    k : Target index.
                """
                return self.apply_controlled(
                    gate.controlled.controlled, i, j, k)
        elif isinstance(gate, Controlled):
            def method(self, i: int, j: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit indices.

                Parameters:
                    i : Control index.
                    j : Target index.
                """
                return self.apply_controlled(gate.controlled, i, j)
        elif isinstance(gate, Box):
            def method(self, i: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit index.

                Parameters:
                    i : Target index.
                """
                return self.apply_controlled(gate, i)
        elif issubclass(gate, Rotation) and issubclass(gate, Controlled):
            def method(self, phi: float, i: int, j: int) -> Circuit:
                """
                Apply :class:`{}` to a circuit given phase and indices.

                Parameters:
                    phi : Phase.
                    i : Control index.
                    j : Target index.
                """
                return self.apply_controlled(gate.controlled(phi), i, j)
        elif issubclass(gate, Rotation):
            def method(self, phi: float, i: int) -> Circuit:
                """
                Apply :class:`{}` to a circuit given phase and target index.

                Parameters:
                    phi : Phase.
                    i : Target index.
                """
                return self.apply_controlled(gate(phi), i)
        method.__doc__ = method.__doc__.format(attr)
        return method
    setattr(Circuit, attr, closure())
