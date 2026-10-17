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
NO_PLANAR_NORMAL_FORM = (
    "A planar law is compared on the nose: a hypergraph is faithful only "
    "from symmetric categories on, and there is no normal form for this "
    "doctrine yet.")
COLOURED_QUOTIENT = (
    "{} is coloured: a hypergraph or a map is symmetric, and a swap is "
    "ill-typed across 0-cells, so it would lose its colours.")
NO_SWAP = "{} is not symmetric, it has no swaps."
NO_BRAID = "{} is not braided, it has no braids."
NO_TWIST = "{} is not balanced, it has no twists."
NO_BRAID_GROUP = (
    "A braid is a box of the hypergraph of a category that is not "
    "symmetric: there is no braid group implementation to compare braid "
    "words with yet.")
NO_CUPS = "{} is not rigid, it has no cups or caps."
NOT_SYMMETRIC = "{} has no swaps to downgrade this map."
NOT_MARKOV = "{} has no copy or discard."
NOT_ACYCLIC = "{} has a directed cycle, its boxes cannot be ordered."

FREE_CURRY = (
    "A free curry is a bubble, which a hypergraph keeps as a box: nothing "
    "evaluates it.")
FREE_FEEDBACK = (
    "A free feedback is a bubble, which a hypergraph keeps as a box: nothing "
    "slides into it.")
RIGID_TYPE_HAS_NO_DAGGER = "Rigid types have no dagger, use pivotal instead."
RIGID_HAS_NO_DAGGER = (
    "Rigid cups and caps have no dagger, use pivotal instead.")
CURRY_HAS_NO_DAGGER = "A curry has no dagger."
FEEDBACK_HAS_NO_DAGGER = (
    "Feedback has no dagger, it would feed its memory back in time.")
MAP_HAS_NO_ADJOINTS = (
    "A map is compact whatever category hosts it, but a host that is not "
    "rigid has no adjoints or curries for the laws to quantify over.")
SECTION_NEEDS_SWAPS = (
    "Decoding a hypergraph orders its boxes with swaps, which a category "
    "with no symmetry does not have.")
MAP_OF_A_CURRY = (
    "The map of a curry rebuilds its inside from the map of it, which can "
    "whisker a scalar on the other side, while a hypergraph compares the "
    "inside of a bubble syntactically.")
MAP_HAS_NO_SPIDERS = (
    "A map has no spiders: it keeps a copy as a box where a hypergraph merges "
    "its wires.")
MISSING_BOXES_AND_OFFSETS = (
    "Decoding needs either boxes and offsets, or the two of them zipped.")
NO_TYPE_PARAMETER = (
    "{} has no type parameter to subscript, declare one as in class C[name].")
NO_SECTION = "{} has no section that is a hypergraph of its category."
NO_FIT = (
    "{} does not fit the boundary {} -> {}: it has another number of inputs "
    "or outputs, or it identifies two positions that the region keeps apart, "
    "e.g. with a wire passing through, which would relate them in every "
    "section.")
NOT_DETERMINISTIC = (
    "The boxes of a Markov category need not be deterministic, so two of "
    "them on related inputs need not have related outputs.")
INDUCTIVE_CONGRUENCE = (
    "Congruence is closed as a least fixpoint, which relates nothing on a "
    "feedback loop: two copies of a cycle are related only coinductively.")
NO_SQUARE = (
    "A spider that merges is not natural along a copy, so a square along "
    "the copy of a memory is not drawn by construction.")
NOT_UNIFORM = (
    "A free trace keeps a loop nothing writes, which a least fixed point "
    "makes its least element, the one value copying preserves: the trace of "
    "the identity on a memory is one such loop, on two copies of it two.")
