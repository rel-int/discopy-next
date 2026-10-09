# -*- coding: utf-8 -*-

"""
discopy error messages.
"""

TYPE_ERROR = "Expected {}, got {} instead."
NOT_COMPOSABLE = "{} does not compose with {}: {} != {}."
NOT_PARALLEL = "Expected parallel arrows, got {} and {} instead."
NOT_GLOBULAR = "Expected a globular box, got boundaries {} -> {} and {} -> {}."
NOT_ATOMIC = "Expected {} of length 1, got length {} instead."
NOT_CONNECTED = "{} is not boundary-connected."
NOT_TRACEABLE = "Cannot trace {} with {}."
NOT_ADJOINT = "{} and {} are not adjoints."
NOT_RIGID_ADJOINT = "{} is not the left adjoint of {}, maybe you meant to use"\
                    " a pivotal type rather than a rigid one?"
MISSING_TYPES_FOR_EMPTY_SUM = "Empty sum needs a domain and codomain."
MATRIX_REPEAT_ERROR = "The reflexive transitive closure is only defined for "\
                      "square boolean matrices."
BOX_IS_MIXED = "Pure boxes can have only digits or only qudits as dom and cod."
LAYERS_MUST_HAVE_A_BOX = "Layers must have at least one box."
NOT_MERGEABLE = "Layers {} and {} cannot be merged."
INTERCHANGER_ERROR = "Boxes {} and {} do not commute."
WRONG_PERMUTATION = "Expected a permutation of length {}, got {}."
ZERO_DISTANCE_CONTROLLED = "Zero-distance controlled gates are ill-defined."
WRONG_DOM = "Expected inside.dom == {}, got {} instead."
WRONG_COD = "Expected inside.cod == {}, got {} instead."
NOT_RIGID = "{} has no cups or caps for the wiring of this map."
NOT_TRACED = "{} has no traces for the cycles of this map."
NOT_SYMMETRIC = "{} has no swaps to downgrade this map."
NOT_ACYCLIC = "{} has a directed cycle, its boxes cannot be ordered."
PERMUTATION_HAS_NO_OFFSET = (
    "A layer with a non-identity permutation has no single offset.")
