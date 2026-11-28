# -*- coding: utf-8 -*-

"""
The Lambek calculus with a relevant modality ``!L*``, i.e. the free
biclosed category of :mod:`~discopy.lambek.categorial` with the bang of
:mod:`linear.monoidal <discopy.linear.monoidal>`: a banged type ``!x`` may be
copied by contraction, read once by dereliction and moved past any type by
exchange, see McPheat, Wijnholds, Sadrzadeh, Correia and Toumi
(`arXiv:2110.10641 <https://arxiv.org/abs/2110.10641>`_).

A term is a kappa term of :mod:`linear.monoidal <discopy.linear.monoidal>`
whose functions are words of exponential type, applied by :class:`Eval`: a
banged variable used twice is copied by :class:`Contraction`, brought where it
is read by :class:`Exchange`, and every other variable is used once and in
order.

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

from discopy.linear import monoidal as bang
from discopy.abc import ClosedMonoidalCategory
from discopy.cat import factory
from discopy.lambek import categorial


@factory
class Ty(bang.Ty, categorial.Ty):
    """
    A type of ``!L*``, a categorial type whose wires may be banged, called
    on a Python function for the kappa abstraction of
    :mod:`linear.monoidal <discopy.linear.monoidal>`.

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


class Box(bang.Box, categorial.Box, Diagram):
    """ A box of ``!L*``, e.g. a word whose type is its codomain. """


class Eval(categorial.Eval, Box):
    """ The evaluation of an exponential type, a box of ``!L*``. """


class Coeval(categorial.Coeval, Box):
    """ The coevaluation of an exponential type, a box of ``!L*``. """


class Curry(categorial.Curry, Box):
    """ The currying of a diagram of ``!L*``. """


class Bang(bang.Bang, Box):
    """ The bang of a diagram of ``!L*``. """


class Dereliction(bang.Dereliction, Box):
    """ The dereliction ``!x -> x`` of ``!L*``. """


class Digging(bang.Digging, Box):
    """ The digging ``!x -> !!x`` of ``!L*``. """


class Contraction(bang.Contraction, Box):
    """ The contraction ``!x -> !x @ !x`` of ``!L*``. """


class Weakening(bang.Weakening, Box):
    """ The weakening ``!x -> 1`` of ``!L*``. """


class Exchange(bang.Exchange, Box):
    """ The exchange ``!x @ y -> y @ !x`` of ``!L*``. """


class Laxator(bang.Laxator, Box):
    """ The laxator ``!x @ !y -> !(x @ y)`` of ``!L*``. """


class Functor(bang.Functor, categorial.Functor):
    """
    A functor from ``!L*`` into a closed monoidal category: the
    exponentials, evaluation and currying go through the closed structure of
    the codomain, and the bang and the structural boxes through its bang when
    it is linear. Into a category that is not closed, e.g. a monoidal one, send
    the underlying monoidal diagram with a :class:`~discopy.monoidal.Functor`,
    which takes them through ``ob_map`` and ``ar_map`` like generators.

    Example
    -------
    Into tensors, which are compact hence closed, ``N >> S`` is the internal
    hom ``N* @ S`` and its evaluation a cup.

    >>> from discopy import tensor
    >>> N, S = Ty('N'), Ty('S')
    >>> F = Functor(
    ...     ob_map={N: tensor.Dim(2), S: tensor.Dim(3)}, ar_map={},
    ...     cod=tensor.Diagram)
    >>> assert F(N >> S) == tensor.Dim(2, 3)
    >>> assert F(Eval(N >> S)).dom == tensor.Dim(2, 2, 3)
    >>> assert F(Eval(N >> S)).cod == tensor.Dim(3)

    >>> from discopy import monoidal
    >>> Functor(ob_map={}, ar_map={}, cod=monoidal.Diagram)
    Traceback (most recent call last):
    ...
    TypeError: A functor from !L* needs a closed monoidal codomain, got ...
    """
    dom = cod = Diagram
    requires = ClosedMonoidalCategory

    def __init__(self, ob_map, ar_map, **params):
        cod = params.get("cod") or type(self).cod
        if not issubclass(cod, self.requires):
            raise TypeError("A functor from !L* needs a closed monoidal "
                            f"codomain, got {cod}.")
        super().__init__(ob_map, ar_map, **params)


class TermBase(bang.TermBase):
    """
    A kappa term of ``!L*``, see :class:`~discopy.linear.monoidal.TermBase`.
    """


class Variable(TermBase, bang.Variable):
    """ A variable of ``!L*``. """


class Constant(TermBase, bang.Constant):
    """ A box of ``!L*`` as a term. """


class Application(TermBase, bang.Application):
    """ An application of ``!L*``, which shares banged variables. """


class Kappa(TermBase, bang.Kappa):
    """ A kappa abstraction of ``!L*``. """


Diagram.box_factory, Diagram.functor_factory = Box, Functor
Diagram.eval_factory, Diagram.coeval_factory = Eval, Coeval
Diagram.curry_factory, Diagram.bang_factory = Curry, Bang
Diagram.dereliction_factory, Diagram.digging_factory = Dereliction, Digging
Diagram.contraction_factory, Diagram.weakening_factory = (
    Contraction, Weakening)
Diagram.exchange_factory, Diagram.laxator_factory = Exchange, Laxator

TermBase.functor = Functor.id(Diagram)
TermBase.variable_factory, TermBase.constant_factory = Variable, Constant
TermBase.application_factory, TermBase.kappa_factory = Application, Kappa
Diagram.term_factory = Ty.term_factory = TermBase
