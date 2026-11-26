semantics
=========

DisCoPy's computational core: concrete categories (functions, matrices,
tensors, constraint satisfaction problems, finite groups, representations of Hopf algebras)
and constructions on categories (opposites, Kleisli categories of monads,
change of base along a monad, streams, parametric maps, lenses,
the Int-construction) and monoidal e-graphs, i.e. e-graphs of diagrams with
sharing.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.python
    discopy.group
    discopy.kleisli
    discopy.monad
    discopy.enriched
    discopy.matrix
    discopy.tensor
    discopy.csp
    discopy.peirce
    discopy.hopf
    discopy.stream
    discopy.op
    discopy.para
    discopy.lens
    discopy.interaction
    discopy.egraph

Drawing: a diagram's layout as a functor into labeled graphs embedded in
the plane, and the backends that render them.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.drawing
    discopy.drawing.drawing
    discopy.drawing.backend
    discopy.drawing.bracketing
    discopy.drawing.typst_ast
    discopy.drawing.widget
    discopy.drawing.nicer
