# -*- coding: utf-8 -*-

"""
discopy error messages.
"""

TYPE_ERROR = "Expected {}, got {} instead."
NOT_COMPOSABLE = "{} does not compose with {}: {} != {}."
NO_COMPOSITE = (
    "{} composes with no functor but the identity, as its maps do not "
    "determine it: apply them in turn instead, i.e. E(D(f)) rather than "
    "(D >> E)(f).")
NOT_PARALLEL = "Expected parallel arrows, got {} and {} instead."
ONLY_REVERSED = "A {} is not sliced, only reversed with [::-1]."
NOT_GLOBULAR = "Expected a globular box, got boundaries {} -> {} and {} -> {}."
NOT_ATOMIC = "Expected {} of length 1, got length {} instead."
NOT_CONNECTED = "{} is not boundary-connected."
NOT_TRACEABLE = "Cannot trace {} with {}."
NOT_UNCURRIABLE = "Cannot uncurry {} of the {} objects in the exponent of {}."
NEGATIVE_STEPS = "Expected a non-negative number of steps, got {} instead."
UNEQUAL_BOUNDARIES = "Expected boundaries of equal length, got {} and {}."
PLANAR_YANKING = "A planar traced category has no braid to yank."
FREE_YANKING = (
    "A free trace is a bubble and a free twist a box: the free balanced "
    "diagrams are not quotiented by the yanking axiom.")
FREE_TWIST = (
    "A free twist is a box: the free balanced diagrams are not quotiented "
    "by the naturality of the twist.")
NOT_FEEDBACK = "Cannot feed {} back into {}, expected {} as its input."
STREAM_SLIDING = (
    "A morphism slides along the feedback of streams only up to extensional "
    "equivalence, see discopy.feedback.")
FEEDBACK_FORMS = (
    "Feed back either a number of objects n or a memory dom, cod, mem.")
NOT_SPLITTABLE = "Cannot split {} atoms off {}."
NOT_ADJOINT = "{} and {} are not adjoints."
NOT_EXPONENTIAL = "Expected {} to have an exponential type, got {}."
NOT_MAPPING_OR_CALLABLE = "Expected a Mapping or a Callable, got {!r}."
NOT_RIGID_ADJOINT = "{} is not the left adjoint of {}, maybe you meant to use"\
                    " a pivotal type rather than a rigid one?"
MISSING_TYPES_FOR_EMPTY_SUM = "Empty sum needs a domain and codomain."
MATRIX_REPEAT_ERROR = "The reflexive transitive closure is only defined for "\
                      "square boolean matrices."
MATRIX_RESOLVENT_ERROR = "The resolvent is only defined for square matrices "\
                         "over a field, i.e. float or complex: a Matrix[int] "\
                         "can be cast to float first."
MATRIX_SINGULAR = "id - D is singular: the trace over a field is partial."
MATRIX_ORDER_ERROR = "Matrices over {} have no pointwise order."
BOX_IS_MIXED = "Pure boxes can have only digits or only qudits as dom and cod."
LAYERS_MUST_HAVE_A_BOX = "Layers must have at least one box."
NOT_MERGEABLE = "Layers {} and {} cannot be merged."
LAYERS_DO_NOT_COMPOSE = "Compose diagrams, not layers: {}."
INTERCHANGER_ERROR = "Boxes {} and {} do not commute."
NO_BRAID_TO_SLIDE = "Box {} at layer {} has no braid to slide through."
WRONG_PERMUTATION = "Expected a permutation of length {}, got {}."
NOT_A_PERMUTATION = "Expected a permutation of range({}), got {}."
NOT_A_SWAP = "Expected the swap [1, 0], got {}."
WRONG_FUNCTION = "Expected a function of length {}, got {}."
NOT_A_FUNCTION = "Expected a function to range({}), got {}."
ZERO_DISTANCE_CONTROLLED = "Zero-distance controlled gates are ill-defined."
REAL_SQRT_IS_SELF_ADJOINT = "The dagger of a real square root is itself."
WRONG_DOM = "Expected inside.dom == {}, got {} instead."
WRONG_COD = "Expected inside.cod == {}, got {} instead."
NOT_FROBENIUS = \
    "{} has no spiders, cups or caps to downgrade this hypergraph."
NOT_RIGID = "{} has no cups or caps for the wiring of this map."
NOT_TRACED = "{} has no traces for the cycles of this map."
DRAWING_DAGGER_LAYOUT = (
    "A drawing is compared on the nose, positions included: composition "
    "aligns wires from the top down and a tensor puts its left boxes first, "
    "so the reflection of a composite or a tensor can be laid out with "
    "other positions or box indices.")
ONLY_SWAP_INVERSE = "Only a swap has an inverse: a function has no dagger."
NO_PLANAR_NORMAL_FORM = (
    "A planar law is compared on the nose: a hypergraph is faithful only "
    "from symmetric categories on, and there is no normal form for this "
    "doctrine yet.")
NOT_BOUNDARY_CONNECTED = (
    "A rigid law is compared on the nose: its sides have a normal form only "
    "when they are boundary-connected, which they need not be.")
COLOURED_QUOTIENT = (
    "{} is coloured: a hypergraph or a map is symmetric, and a swap is "
    "ill-typed across 0-cells, so it would lose its colours.")
NO_VALUES = "{} has no sample of values, see testable.values."
NO_DAGGER = "There is no dagger here, so the dagger laws do not apply."
SWAP_INVOLUTION = (
    "The inverse of a swap is the swap the other way, see swap_inverse.")
PAIR_LENGTH = (
    "A pair of types has no single number of atoms for a trace or a curry "
    "to count.")
PAIR_TYPES = "A pair of types is checked as the boundary of its diagrams."
PARA_REPARAM = (
    "A parametric map is equal only up to the reparametrisations that "
    "permute its parameters, which its hypergraph compares in order.")
NO_ROTATION = (
    "There is no rotation of a morphism here, so its laws do not apply.")
PARA_CURRY = "A closed category curries no map with coparameters."
PARA_CURRY_COPAR = (
    "A closed category cannot curry a map with coparameters {}, as its "
    "exponential would swallow them.")
LENS_NATURALITY = (
    "A Markov lens is a unit and associative only up to the naturality of "
    "copy and discard, which a hypergraph does not decide.")
LENS_BRAID_NATURALITY = (
    "A Markov braid copies its input for its put, so a Markov lens is "
    "natural in it only up to the naturality of copy, which a hypergraph "
    "does not decide.")
PRISM_NATURALITY = (
    "A prism is a unit and associative only up to the naturality of merge "
    "and noise, which a hypergraph does not decide.")
PRISM_BRAID_NATURALITY = (
    "A prism braid merges its build with its residual, so a prism is "
    "natural in it only up to the naturality of merge, which a hypergraph "
    "does not decide.")
NO_SWAP = "{} is not symmetric, it has no swaps."
NO_BRAID = "{} is not braided, it has no braids."
NO_TWIST = "{} is not balanced, it has no twists."
NO_BRAID_GROUP = (
    "A braid is a box of the hypergraph of a category that is not "
    "symmetric: there is no braid group implementation to compare braid "
    "words with yet.")
NO_CUPS = "{} is not rigid, it has no cups or caps."
NO_SUMS = "{} has no sums, it encodes a single diagram."
NO_PHASES = "{} is not a hypergraph category, its spiders have no phases."
NOT_INVERTIBLE = "{} has no inverse, it is not sqrt(2) ** n times a phase."
NOT_SYMMETRIC = "{} has no swaps to downgrade this map."
NOT_MARKOV = "{} has no copy or discard."
NOT_COMARKOV = "{} has no merge or noise."
NOT_A_DAGGER = "{} has no dagger: {} is not a dagger category."
NO_SUCH_SPIDER = "The only spiders of {} are its {}."
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
FEEDBACK_HAS_NO_DAGGER = (
    "Feedback has no dagger and no rotation, "
    "either would feed its memory back in time.")
MAP_HAS_NO_ADJOINTS = (
    "A map is compact whatever category hosts it, but a host that is not "
    "rigid has no adjoints or curries for the laws to quantify over.")
MAP_HAS_NO_SPIDERS = (
    "A map has no spiders: it keeps a copy or a merge as a box where a "
    "hypergraph merges its wires.")
STR_SPLITS_LAYERS = (
    "A string tensors the boxes of a layer one after the other, so that a "
    "layer of several boxes, as a hypergraph decodes to, reads back as "
    "several layers, equal to it only up to interchange.")
NO_TYPE_PARAMETER = (
    "{} has no type parameter to subscript, declare one as in class C[name].")
NOT_PROGRESSIVE = "No line sweeps the map down meeting each box with its "\
    "inputs side by side, past the cut {}."
NOT_DOWNWARD = "{} has cups, caps or loops: its wires do not all go down."
NAT_HAS_NO_ADJOINTS = (
    "The objects of a cospan of finite sets are numbers, with no adjoints "
    "for the laws and rules of rigid and closed categories to quantify over.")
FREE_SCALAR = "A free scalar is a box: it slides on the map of a diagram, "\
    "not on the nose."
NOT_LEFT_MONOGAMOUS = "{} is not left-monogamous."
NOT_RIGHT_MONOGAMOUS = "{} is not right-monogamous."
NOT_CAUSAL = "{} is not causal."
NOT_BIJECTIVE = "{} is not bijective: a spider has neither zero nor two legs."
NOT_AN_INVOLUTION = \
    "Expected a fixpoint-free involution, got edges {} instead."
NOT_PLUGGABLE = "Cannot plug {} into a map with codomain {}: expected a box "\
                "with one input and two outputs, and a map with one output."
NOT_AN_INDEX = "Expected an index in range({}), got {} instead."
WRONG_NUMBER_OF_WIRES = "Expected {} wires for {}, got {} instead."
WRONG_NUMBER_OF_BOUNDARIES = \
    "Expected one boundary of wires for each of {} boxes, got {} instead."
TERM_IS_A_BOX = \
    "A term is a box: the laws and rules of its category are the diagram's."
MISSING_BOXES_AND_OFFSETS = (
    "Decoding needs either boxes and offsets, or the two of them zipped.")
OVERLAPPING_BOXES = (
    "{} at offset {} overlaps the box on its left, which ends at {}.")
CURRY_HAS_NO_DAGGER = "A curry has no dagger."
BANG_HAS_NO_DAGGER = (
    "{} has no dagger: !L* has no rule from its codomain back to its domain.")
LAXATOR_ARITY = (
    "A laxator takes two types, or none for the unit, got {}.")
NOT_BANGED = (
    "{} is not banged: only a type !x is copied by contraction and "
    "discarded by weakening.")
NOT_SHARED = "Expected banged variables to share, got {}."
FREE_BANG = (
    "A free bang is a bubble and its structure boxes, which a hypergraph "
    "keeps as boxes: nothing cancels, merges or slides through them.")
NAME_CLASH = "{!r} names two things in the string of {}."
WRONG_SIDE = "Expected left={} for {}, got left={} instead."
NOT_JSON_SERIALISABLE = "Object of type {} is not JSON serialisable."
NO_TREE_CALLABLE = "{} has no tree: its {} is a Python callable, which "\
    "JSON cannot hold."
NO_TREE_DRAWING = "A drawing has no tree: it is the layout of a diagram, "\
    "which to_drawing recomputes."
NO_TREE_PARAMS = "A tensor functor with params or a contractor has no "\
    "tree: they go to the einsum of a backend verbatim, which a file should "\
    "not reach."
NO_TREE_STREAM = "A stream has no tree: its later is what a Python callable "\
    "returns or, for a constant stream, itself, which JSON cannot hold."
NO_SUCH_TYPE = "No type named {!r} in builtins or numpy, so the name of the "\
               "class it parameterises cannot be resolved."
NO_SUCH_CLASS = "No class of DisCoPy is named {!r}."

NO_COMMON_BOUNDARY = "A bubble needs a {} given when its args do not "\
    "all share one, they have [{}]."
NOT_REWRITABLE = (
    "{} is not rewritable: double-pushout rewriting is sound over a "
    "hypergraph category, and over a symmetric one for monogamous acyclic "
    "hypergraphs.")
NOT_A_HOLE = "Cannot plug {} into {}: its boundary is not the hole's."
NOT_LEFT_LINEAR = (
    "{} is not left-linear: rewriting in place needs a left side with "
    "boxes and with its boundary wires on distinct spiders.")
NOT_GRAPHLIKE = "{} is not a green or red spider, a Hadamard or a scalar."
NOT_SHARING = (
    "{} is not a diagram with sharing, i.e. a left-monogamous acyclic "
    "hypergraph: a spider has no producer or several, or a box depends on "
    "itself.")
NOT_CARTESIAN = (
    "{} is not cartesian, i.e. deterministic and affine: an effect is "
    "context, which a rewrite keeps, as a section that chooses a producer "
    "for each class would decorrelate a stochastic one.")
NOT_A_BOUNDARY = "{} does not have the types of the spiders {} -> {}."
NOT_A_REGION = (
    "The boxes {} do not compute the spiders {} -> {}: they cannot all fire "
    "in some order from the classes of the domain, or do not reach those of "
    "the codomain.")
NOT_A_PATTERN = (
    "{} is not a pattern, i.e. a cartesian diagram with sharing whose "
    "codomain is produced by its boxes: a variable alone matches anything.")
NOT_A_RULE = (
    "{} = {} is not a rule: its sides are cartesian diagrams of one type, the "
    "left a pattern and the right reading only the inputs the left reads.")

LAYOUT_LENGTH = "A layout of {} coordinates was given, {} expected."
LAYOUT_NOT_FEASIBLE = "This layout does not draw the same diagram."
STRETCH_LENGTHS = (
    "Give a scale for each gap between columns and between rows, and for "
    "the width and the height of each box.")
NOT_ONE_OUTPUT = "Expected exactly one output, got {}: a term has one."
NOT_APPLICABLE = "{} of stack {} does not apply to {} of stack {} and type {}."
NOT_A_BLOCK = "Expected a block of variables next to the stack of {}, got {}."
STATEMENTS_DO_NOT_COMMUTE = (
    "A statement binding several outputs or none stays in a normal form, "
    "and two of them do not commute.")
NO_HIGHEST_LEVEL = "Expected terms of one level above the others, got {}."
NOT_DISTINCT = "Expected distinct variables to bind in {}, got {}."
NOT_FREE = "Expected distinct free variables of {}, got {}."
NOT_THE_VARIABLE = "Expected the free variable of {}, with no stack, got {}."
NO_OFFSET = (
    "Expected an offset for the abstraction of no variable in {}, which "
    "has no stack to say where it stands.")
WRONG_OFFSET = "Expected offset {} for the abstraction of {} in {}, got {}."
