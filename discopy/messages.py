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
PLANAR_YANKING = "A planar traced category has no braid to yank."
FREE_YANKING = (
    "A free trace is a bubble and a free twist a box: the free balanced "
    "diagrams are not quotiented by the yanking axiom.")
NOT_FEEDBACK = "Cannot feed {} back into {}, expected {} as its input."
STREAM_SLIDING = (
    "A morphism slides along the feedback of streams only up to extensional "
    "equivalence, see discopy.feedback.")
FEEDBACK_FORMS = (
    "Feed back either a number of objects n or a memory dom, cod, mem.")
NOT_SPLITTABLE = "Cannot split {} atoms off {}."
NOT_ADJOINT = "{} and {} are not adjoints."
NOT_MAPPING_OR_CALLABLE = "Expected a Mapping or a Callable, got {!r}."
NOT_RIGID_ADJOINT = "{} is not the left adjoint of {}, maybe you meant to use"\
                    " a pivotal type rather than a rigid one?"
MISSING_TYPES_FOR_EMPTY_SUM = "Empty sum needs a domain and codomain."
MATRIX_TWO_DTYPES = "Matrix class cannot be indexed twice."
MATRIX_REPEAT_ERROR = "The reflexive transitive closure is only defined for "\
                      "square boolean matrices."
BOX_IS_MIXED = "Pure boxes can have only digits or only qudits as dom and cod."
LAYERS_MUST_HAVE_A_BOX = "Layers must have at least one box."
NOT_MERGEABLE = "Layers {} and {} cannot be merged."
INTERCHANGER_ERROR = "Boxes {} and {} do not commute."
WRONG_PERMUTATION = "Expected a permutation of length {}, got {}."
ZERO_DISTANCE_CONTROLLED = "Zero-distance controlled gates are ill-defined."
REAL_SQRT_IS_SELF_ADJOINT = "The dagger of a real square root is itself."
HAS_NO_ATTRIBUTE = "{!r} object has no attribute {!r}"
WRONG_DOM = "Expected inside.dom == {}, got {} instead."
WRONG_COD = "Expected inside.cod == {}, got {} instead."
COMPLEX_TYPE_HAS_NO_ATTR = "{!r} object of length != 1 has no attribute {!r}"
NOT_FROBENIUS = \
    "{} has no spiders, cups or caps to downgrade this hypergraph."
NOT_RIGID = "{} has no cups or caps for the wiring of this map."
NOT_TRACED = "{} has no traces for the cycles of this map."
NO_SWAP = "{} is not symmetric, it has no swaps."
NO_BRAID = "{} is not braided, it has no braids."
NO_TWIST = "{} is not balanced, it has no twists."
NO_CUPS = "{} is not rigid, it has no cups or caps."
COLOURED_QUOTIENT = (
    "{} is coloured: a hypergraph or a map is symmetric, and a swap is "
    "ill-typed across 0-cells, so it would lose its colours.")
NOT_SYMMETRIC = "{} has no swaps to downgrade this map."
NOT_MARKOV = "{} has no copy or discard."
NOT_A_DAGGER = "{} has no dagger: {} is not a dagger category."
NO_SUCH_SPIDER = "The only spiders of {} are its {}."
NOT_ACYCLIC = "{} has a directed cycle, its boxes cannot be ordered."
NOT_LEFT_MONOGAMOUS = "{} is not left-monogamous."
NOT_RIGHT_MONOGAMOUS = "{} is not right-monogamous."
NOT_CAUSAL = "{} is not causal."
WRONG_NUMBER_OF_WIRES = "Expected {} wires for {}, got {} instead."
WRONG_NUMBER_OF_BOUNDARIES = \
    "Expected one boundary of wires for each of {} boxes, got {} instead."
TERM_IS_A_BOX = \
    "A term is a box: the laws of its category are the diagram's."
MISSING_BOXES_AND_OFFSETS = (
    "Decoding needs either boxes and offsets, or the two of them zipped.")
NO_TYPE_PARAMETER = (
    "{} has no type parameter to subscript, declare one as in "
    "class C(NamedGeneric['name']).")
CURRY_HAS_NO_DAGGER = "A curry has no dagger."
NOT_JSON_SERIALISABLE = "Object of type {} is not JSON serialisable."
NO_SUCH_TYPE = "No type named {!r} in builtins or numpy, so the name of the "\
               "class it parameterises cannot be resolved."

NO_COMMON_BOUNDARY = "A bubble needs a {} given when its args do not "\
    "all share one, they have [{}]."
