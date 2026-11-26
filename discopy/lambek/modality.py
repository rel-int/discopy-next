# -*- coding: utf-8 -*-

"""
The Lambek calculus with a relevant modality ``!L*``, i.e. the free biclosed
category of :mod:`discopy.lambek.categorial` with the bang of
:mod:`discopy.bang`: a banged type ``!x`` may be copied by contraction,
read once by dereliction and moved past any type by exchange, see
McPheat, Wijnholds, Sadrzadeh, Correia and Toumi
(`arXiv:2110.10641 <https://arxiv.org/abs/2110.10641>`_).

A term is a kappa term of :mod:`discopy.bang` whose functions are words
of exponential type, applied by :class:`Eval`: a banged variable used twice
is copied by :class:`Contraction`, brought where it is read by
:class:`Exchange`, and every other variable is used once and in order.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Diagram
    Box
    Eval
    Coeval
    Curry
    Bang
    Dereliction
    Digging
    Contraction
    Weakening
    Exchange
    Laxator
    Functor
    TermBase
    Variable
    Constant
    Application
    Kappa

Example
-------
*John sleeps. He snores.*: the subject ``John : !N`` is copied, one copy
read by dereliction for ``sleeps : N\\S``, the other passed to the anaphor
``He : !N\\N``, and the full stop puts the two sentences side by side.

>>> N, S = Ty('N'), Ty('S')
>>> John = Box('John', Ty(), N.bang())
>>> sleeps, snores = Box('sleeps', Ty(), N >> S), Box('snores', Ty(), N >> S)
>>> He, stop = Box('He', Ty(), N.bang() >> N), Box('.', S @ S, S @ S)
>>> term = N.bang()(lambda john: stop(
...     Eval(N >> S)(Dereliction(N)(john), sleeps()),
...     Eval(N >> S)(Eval(N.bang() >> N)(john, He()), snores())))(John())
>>> diagram = term.eval()
>>> assert diagram.dom == Ty() and diagram.cod == S @ S
>>> assert Contraction(N) in diagram.boxes
"""

from __future__ import annotations

from discopy import bang, monoidal
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.lambek import categorial
from discopy.utils import export, get_origin


@factory
class Ty(bang.Ty, categorial.Ty):
    """
    A type of ``!L*``, a categorial type whose wires may be banged, called
    on a Python function for the kappa abstraction of :mod:`discopy.bang`.

    Example
    -------
    >>> N, S = Ty('N'), Ty('S')
    >>> likes = ((N.bang() >> S).bang() << N).bang()
    >>> print(likes)
    ((N.bang() >> S).bang() << N).bang()
    >>> assert eval(str(likes)) == likes
    """


@factory
class Diagram(bang.Diagram, categorial.Diagram):
    """
    A diagram of ``!L*``, i.e. both linear and biclosed.

    Parameters:
        inside (Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty


@Diagram.extend
class Functor(bang.Functor, categorial.Functor):
    """
    A functor from ``!L*``: the bang and the structural boxes go through the
    bang of the codomain when it is linear, and the exponentials, evaluation
    and currying through its closed structure when it is closed. Into any
    other category, e.g. the photonic diagrams of :mod:`discopy.optyx`, they
    go through ``ob_map`` and ``ar_map`` like generators.

    Example
    -------
    >>> from discopy import tensor
    >>> N, S = Ty('N'), Ty('S')
    >>> F = Functor(
    ...     ob_map=lambda x: tensor.Dim(4 if x.is_exp else 2),
    ...     ar_map={Eval(N >> S): tensor.Box('ev', tensor.Dim(2, 4), 2)},
    ...     cod=tensor.Diagram)
    >>> assert F(Eval(N >> S)).dom == tensor.Dim(2, 4)
    """
    dom = cod = Diagram

    @property
    def has_exp(self) -> bool:
        """ Whether the codomain is a closed category. """
        return issubclass(get_origin(self.cod), ClosedCategory)

    def __call__(self, other):
        if not self.has_exp and isinstance(other, (
                categorial.Over, categorial.Under)):
            return monoidal.Functor.__call__(self, other)
        if not self.has_exp and isinstance(other, (
                categorial.Diagram.Eval, categorial.Diagram.Coeval,
                categorial.Diagram.Curry)):
            return self.image(other)
        return super().__call__(other)


class TermBase(bang.TermBase):
    """ A kappa term of ``!L*``, see :class:`discopy.bang.TermBase`. """


class Variable(TermBase, bang.Variable):
    """ A variable of ``!L*``. """


class Constant(TermBase, bang.Constant):
    """ A box of ``!L*`` as a term. """


class Application(TermBase, bang.Application):
    """ An application of ``!L*``, which shares banged variables. """


class Kappa(TermBase, bang.Kappa):
    """ A kappa abstraction of ``!L*``. """


TermBase.functor = Functor.id(Diagram)
TermBase.Variable, TermBase.Constant = Variable, Constant
TermBase.Application, TermBase.Kappa = Application, Kappa
Diagram.TermBase = Ty.TermBase = TermBase

export(__name__)
