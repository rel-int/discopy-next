"""
The language in which a category states its rules, generators and
axioms: two type aliases and a handful of empty generic classes, read
by Python and by typecheckers alike.

A sequent is the signature of a method on an abstract base class of
:mod:`discopy.abc`: its :pep:`695` type parameter list is the context,
each parameter one variable with its sort as the bound — ``A: Obj[C0]``
an object, ``X: Atom[C0]`` an atomic one, ``N: Count`` a number of
repetitions — its parameters the premises and its return annotation
the conclusion: ``f: Hom[C1, A, B]`` a morphism between two sides and
``x: Obj[C0, p]`` an object standing for the pattern ``p``.

.. code-block:: python

    def tensor[A: Obj[C0], B: Obj[C0], C: Obj[C0], D: Obj[C0]](
            self: Hom[C1, A, B], other: Hom[C1, C, D]
    ) -> Hom[C1, Tensor[A, C], Tensor[B, D]]:
        ...

Nothing here reads an annotation. Python evaluates the signature,
lazily by :pep:`649`, into the objects a typechecker sees:
``Hom[C1, A, B]`` is a generic alias with ``Hom`` as ``__origin__`` and
the type parameters ``C1, A, B`` as ``__args__``, a former such as
``Tensor[A, C]`` is a generic alias of the class :class:`Tensor`, and a
variable is a :class:`typing.TypeVar` with its sort as ``__bound__``.
:mod:`discopy.sequent` matches these objects against goals.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Tensor
    Unit
    L
    R
    D
    Over
    Under
    Repeat
    Count
"""

from typing import Annotated, Literal


type Obj[Coarse, Fine = None, Size = None] = Annotated[Coarse, Fine, Size]
""" The premise ``x: Obj[T, p]`` of an object standing for a pattern
``p``, beside its coarse type ``T``, and the bound ``A: Obj[C0]`` of
an object variable: a typechecker reads it as ``T``. The size, when
given, is the number of wires: ``Literal[n]`` or a variable ``N:
Count``, e.g. ``M: Obj[C0, None, N]``. """

type Atom[Coarse, Fine = None] = Obj[Coarse, Fine, Literal[1]]
""" An object of size one, i.e. a single wire: the premise ``x:
Atom[T, p]`` and the bound ``X: Atom[C0]``. """

type Hom[Coarse, Dom, Cod] = Annotated[Coarse, Dom, Cod]
""" The premise or conclusion ``Hom[C1, dom, cod]`` of a morphism
between two patterns: a typechecker reads it as ``C1``. """


class Tensor[*Ts]:
    """ The tensor ``Tensor[A, B, ...]`` of two or more patterns. """


class Unit[T]:
    """ The unit ``Unit[C0]`` of a monoid of objects. """


class L[T]:
    """ The left adjoint ``L[A]`` of a pattern. """


class R[T]:
    """ The right adjoint ``R[A]`` of a pattern. """


class D[T]:
    """ The delay ``D[A]`` of a pattern by one time step. """


class Over[A, B]:
    """ The exponential ``Over[A, B]``, i.e. ``A << B``. """


class Under[A, B]:
    """ The exponential ``Under[A, B]``, i.e. ``A >> B``. """


class Repeat[X, N]:
    """ An atomic pattern ``X`` repeated ``N`` times, for the legs of a
    spider, ``N`` a variable of sort :class:`Count`. """


class Count:
    """ The bound ``N: Count`` of a number of repetitions. """
