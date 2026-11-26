# -*- coding: utf-8 -*-

"""
The free Markov category with a bang, i.e. a Markov category whose banged
objects contract and weaken by their copy and discard: the
:class:`Contraction` of ``!x`` is ``Copy(!x)`` and its :class:`Weakening` is
``Discard(!x)``, so the comonoid that linear logic puts on a banged object
is the one every object of a Markov category already has.

There is no default bang. The identity is a linear exponential comonad only
in the cartesian case, where every morphism is deterministic, i.e. commutes
with the copies, as the morphisms of a bang do: in a Markov category with
proper stochastic morphisms, ``!f = f`` would make every ``f`` copyable.
A bang of a semantic category is a choice of its own, e.g. that of a
representable Markov category :cite:p:`FritzEtAl23`, whose distribution
functor is right adjoint to the inclusion of the deterministic morphisms by
a symmetric monoidal adjunction (their Proposition 3.15), or the Fock
space of
:mod:`discopy.optyx`, see the notebook :doc:`/notebooks/optyx/linear_logic`;
neither is built here.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Layer
    Diagram
    Box
    Function
    Copy
    Discard
    Permutation
    Swap
    Sum
    Bang
    Structural
    Dereliction
    Digging
    Exchange
    Laxator
    Functor

Example
-------
>>> x, y = Ty('x'), Ty('y')
>>> f = Box('f', x, y)
>>> assert Diagram.contraction(y) == Copy(y.bang())
>>> assert Diagram.weakening(y) == Discard(y.bang())
>>> twice = f.bang() >> Diagram.contraction(y)\\
...     >> Dereliction(y) @ Dereliction(y)
>>> assert twice.dom == x.bang() and twice.cod == y @ y
"""

from __future__ import annotations

from discopy import markov
from discopy.cat import factory
from discopy.linear import symmetric as swaps
# pylint: disable-next=unused-import
from discopy.linear.monoidal import Ty, BangOb  # noqa: F401


class Layer(markov.Layer, swaps.Layer):
    """ A Markov linear layer, with permutations as plumbing. """


@factory
class Diagram(markov.Diagram, swaps.Diagram):
    """
    A Markov linear diagram is a Markov diagram and a symmetric linear
    diagram, whose :meth:`contraction` and :meth:`weakening` are the copy and
    the discard of a banged object.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer
    term_factory = swaps.Diagram.term_factory

    @classmethod
    def contraction(cls, x: Ty) -> Diagram:
        """
        The contraction ``!x -> !x @ !x`` is the copy of ``!x``.

        Parameters:
            x : The type whose bang is contracted.
        """
        return cls.copy(x.bang())

    @classmethod
    def weakening(cls, x: Ty) -> Diagram:
        """
        The weakening ``!x -> 1`` is the discard of ``!x``.

        Parameters:
            x : The type whose bang is weakened.
        """
        return cls.discard(x.bang())


class Box(markov.Box, swaps.Box, Diagram):
    """
    A Markov linear box is a Markov box and a symmetric linear box.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Function(markov.Function, Box):
    """ A function in a Markov linear diagram, see :class:`markov.Function`.
    """


class Copy(markov.Copy, Function):
    """ The copy of a type, see :class:`markov.Copy`. """


class Discard(markov.Discard, Copy):
    """ The discard of a type, see :class:`markov.Discard`. """


class Permutation(markov.Permutation, swaps.Permutation, Box):
    """ A permutation in a Markov linear diagram. """


Function.register(Permutation)


class Swap(Permutation, markov.Swap, swaps.Swap, Box):
    """ A swap in a Markov linear diagram. """


class Sum(markov.Sum, swaps.Sum, Box):
    """ A sum of Markov linear diagrams. """


class Bang(swaps.Bang, Box):
    """ The bang of a Markov linear diagram, a bubble with no default. """


class Structural(swaps.Structural, Box):
    """ A structural box of a Markov linear category. """


class Dereliction(swaps.Dereliction, Structural):
    """
    The dereliction ``!x -> x``, see
    :class:`~discopy.linear.symmetric.Dereliction`.
    """


class Digging(swaps.Digging, Structural):
    """
    The digging ``!x -> !!x``, see
    :class:`~discopy.linear.symmetric.Digging`.
    """


class Exchange(swaps.Exchange, Structural):
    """
    The exchange ``!x @ y -> y @ !x``, see
    :class:`~discopy.linear.symmetric.Exchange`.
    """


class Laxator(swaps.Laxator, Structural):
    """
    The laxator ``!x @ !y -> !(x @ y)``, see
    :class:`~discopy.linear.symmetric.Laxator`.
    """


class Functor(markov.Functor, swaps.Functor):
    """
    A Markov linear functor is a Markov functor and a linear functor.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to
            :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


Id = Diagram.id
Diagram.box_factory = Box
Diagram.sum_factory = Sum
Diagram.function_factory = Function
Diagram.copy_factory = Copy
Diagram.discard_factory = Discard
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.bang_factory = Bang
Diagram.dereliction_factory = Dereliction
Diagram.digging_factory = Digging
Diagram.exchange_factory = Exchange
Diagram.laxator_factory = Laxator
Diagram.functor_factory = Functor

Bubble = Box.from_args
