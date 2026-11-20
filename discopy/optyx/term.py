"""
Linear terms with bangs compiled to optical protocols: a :class:`Term` is a
diagram of :mod:`discopy.bang`, the free linear category, on the dual-rail
:data:`qubit` and its bang, the bosonic Fock space of its two modes.
:meth:`Term.to_channel` compiles it to a :class:`channel.Diagram`:

- a :class:`Box` on one qubit holds the unitary of one photon in two modes,
  and its image is that gate on a dual-rail qubit, heralded,
- its bang ``f.bang()`` is the linear-optical gate of that unitary acting
  on every photon at once, its second quantisation,
- :class:`Contraction` is the W node of ZW on each mode, which shares the
  photons of a mode between two copies, :class:`Weakening` post-selects the
  vacuum and :class:`Dereliction` heralds exactly one photon, reading it back
  as a dual-rail qubit,
- :class:`Exchange` is the swap.

:class:`Digging` and :class:`Laxator` have no such image: the Fock space of
a Fock space is not a set of modes, and ``!x @ !y`` is the Fock space of
``x ⊕ y``, not of ``x @ y``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Term
    Box
    Bang
    Dereliction
    Digging
    Contraction
    Weakening
    Exchange
    Laxator

Example
-------
The non-linear term ``lambda x: (H(x), H(x))`` uses its argument twice: its
argument is banged, the bang of ``H`` is copied by contraction and each copy
is read once by dereliction. On one photon in each mode, the bang of ``H`` is
a Hong-Ou-Mandel beam splitter and the two heralded copies are entangled.

>>> import numpy as np
>>> from discopy.optyx import photonic
>>> H = Box("H", qubit, qubit, data=np.array([[1, 1], [1, -1]]) / 2 ** .5)
>>> @Term.from_callable(qubit.bang(), qubit @ qubit)
... def copy_H(x):
...     y, z = Contraction(qubit)(H.bang()(x))
...     return Dereliction(qubit)(y), Dereliction(qubit)(z)
>>> protocol = photonic.Create(1, 1) >> copy_H.to_channel()
>>> bell = protocol.get_kraus().to_tensor().eval().array
>>> assert np.allclose(bell, [[1, 0], [0, -1]])
"""

from __future__ import annotations

import numpy as np

from discopy import bang
from discopy.cat import factory
from discopy.optyx import channel, photonic
from discopy.optyx.core import diagram, zw
from discopy.optyx.utils.misc import hashable


@factory
class Ty(bang.Ty):
    """
    A type of terms: a tensor of :data:`qubit` and its bang.
    """
    def to_channel(self) -> channel.Ty:
        """
        The image of a type in optyx: a qubit is a dual-rail qubit and its
        bang the two modes it is encoded in.

        Example
        -------
        >>> assert (qubit @ qubit.bang()).to_channel()\\
        ...     == channel.qubit @ channel.qmode @ channel.qmode
        """
        images = {qubit: channel.qubit,
                  qubit.bang(): channel.qmode @ channel.qmode}
        wires = [self[i:i + 1] for i in range(len(self))]
        for wire in wires:
            if wire not in images:
                raise NotImplementedError(
                    f"{wire} has no image in optyx, only {qubit} "
                    f"and {qubit.bang()} do.")
        return channel.Ty().tensor(*(images[wire] for wire in wires))


qubit = Ty("qubit")


@factory
class Term(bang.Diagram):
    """
    A term on qubits with bangs, i.e. a diagram of :mod:`discopy.bang` whose
    boxes hold the unitary of one photon in the two modes of a qubit.

    Parameters:
        inside(Layer) : The layers inside the term.
        dom (Ty) : The domain of the term, i.e. its input.
        cod (Ty) : The codomain of the term, i.e. its output.
    """
    ob = Ty

    def to_channel(self) -> channel.Diagram:
        """
        Compile a term to an optical protocol, sending each box to its own
        ``to_channel``.

        Example
        -------
        >>> U = Box("U", qubit, qubit, data=np.array([[0, 1], [1, 0]]))
        >>> term = Contraction(qubit) >> U.bang() @ Weakening(qubit)
        >>> assert term.to_channel() == Contraction(qubit).to_channel()\\
        ...     >> U.bang().to_channel() @ Weakening(qubit).to_channel()
        """
        return bang.Functor(
            ob_map=Ty.to_channel, ar_map=lambda box: box.to_channel(),
            dom=Term, cod=channel.Diagram)(self)


class Box(bang.Box, Term):
    """
    A box on one qubit, whose ``data`` is the unitary of one photon in its
    two modes: its image is that gate on the dual-rail qubit, heralded.

    Example
    -------
    >>> U = Box("U", qubit, qubit, data=np.array([[0, 1], [1, 0]]))
    >>> print(U.to_channel())
    DualRail(1) >> U >> ε
    >>> assert U == Box("U", qubit, qubit, data=np.array([[0, 1], [1, 0]]))
    """
    def setoid(self):
        """ The setoid of a box, with its array as a hashable. """
        return super().setoid()[:-1] + (hashable(self.data), )

    def to_channel(self) -> channel.Diagram:
        if self.dom != qubit or self.cod != qubit:
            raise NotImplementedError(
                f"{self} has no image in optyx, only a box on one qubit has.")
        gate = photonic.Gate(np.transpose(self.data), 2, 2, self.name)
        return photonic.DualRail(1) >> gate\
            >> Dereliction(qubit).to_channel()


class Bang(bang.Bang, Box):
    """
    The bang of a term on one qubit made of boxes, whose image is the
    linear-optical gate of the product of their unitaries, acting on every
    photon at once.

    Parameters:
        arg : The term to bang.

    Example
    -------
    >>> X = Box("X", qubit, qubit, data=np.array([[0, 1], [1, 0]]))
    >>> Z = Box("Z", qubit, qubit, data=np.array([[1, 0], [0, -1]]))
    >>> assert np.allclose(
    ...     (X >> Z).bang().to_channel().array, (Z.data @ X.data).T)
    """
    def to_channel(self) -> channel.Diagram:
        if self.arg.dom != qubit or any(
                box.dom != qubit or box.cod != qubit
                for box in self.arg.boxes):
            raise NotImplementedError(
                f"{self} has no image in optyx, only the bang of boxes "
                "on one qubit has.")
        unitary = np.eye(2)
        for box in self.arg.boxes:
            unitary = box.data @ unitary
        return photonic.Gate(unitary.T, 2, 2, f"!{self.arg}")


class Dereliction(bang.Dereliction, Box):
    """
    The dereliction of :mod:`discopy.bang`, which heralds one photon in the
    two modes of a qubit.

    Example
    -------
    >>> assert Dereliction(qubit).to_channel().dom\\
    ...     == channel.qmode @ channel.qmode
    """
    def to_channel(self) -> channel.Diagram:
        return channel.Channel(
            "ε", diagram.DualRail(is_dagger=True),
            self.dom.to_channel(), self.cod.to_channel())


class Digging(bang.Digging, Box):
    """
    The digging of :mod:`discopy.bang`, which has no image in optyx.
    """


class Contraction(bang.Contraction, Box):
    """
    The contraction of :mod:`discopy.bang`, the W node on each mode.

    Example
    -------
    >>> assert len(Contraction(qubit).to_channel().cod) == 4
    """
    def to_channel(self) -> channel.Diagram:
        return channel.Channel(
            "Δ", zw.W(2) @ zw.W(2) >> diagram.mode @ zw.SWAP @ diagram.mode,
            self.dom.to_channel(), self.cod.to_channel())


class Weakening(bang.Weakening, Box):
    """
    The weakening of :mod:`discopy.bang`, which selects the vacuum.

    Example
    -------
    >>> assert not Weakening(qubit).to_channel().cod
    """
    def to_channel(self) -> channel.Diagram:
        return channel.Channel(
            "e", zw.Select(0, 0),
            self.dom.to_channel(), self.cod.to_channel())


class Exchange(bang.Exchange, Box):
    """
    The exchange of :mod:`discopy.bang`, the swap.

    Example
    -------
    >>> assert Exchange(qubit, qubit).to_channel()\\
    ...     == channel.Diagram.swap(channel.qmode ** 2, channel.qubit)
    """
    def to_channel(self) -> channel.Diagram:
        swap = channel.Diagram.swap(
            self.x.bang().to_channel(), self.y.to_channel())
        return swap.dagger() if self.is_dagger else swap


class Laxator(bang.Laxator, Box):
    """
    The laxator of :mod:`discopy.bang`, which has no image in optyx.
    """


Id = Term.id
Term.box_factory = Box
Term.bang_factory = Bang
Term.dereliction_factory = Dereliction
Term.digging_factory = Digging
Term.contraction_factory = Contraction
Term.weakening_factory = Weakening
Term.exchange_factory = Exchange
Term.laxator_factory = Laxator
