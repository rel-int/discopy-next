# -*- coding: utf-8 -*-

"""
A categorial grammar is a free biclosed category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Over
    Under
    TermBase
    Constant
    Variable
    Abstraction
    FA
    BA
    TypeRaising
    FTR
    BTR
    BinaryTerm
    FC
    BC
    FX
    BX
    Diagram
    Box
    Eval
    Coeval
    Curry
    Word
    ForwardCrossedComposition
    BackwardCrossedComposition
    Bubble
    Functor
    CMap

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        cat2ty
        tree2diagram
        combine
        parse

.. admonition:: Bobcat

    .. autosummary::
        :template: class.rst
        :nosignatures:
        :toctree:

        Derivation
        Parser

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        normalise
        structure
        render
        schemata
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

from discopy import biclosed, cmap, messages
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.lambek import thue
from discopy.utils import (
    BinaryBoxConstructor,
    AxiomError,
    factory_key,
    factory_name,
    from_tree,
    get_origin,
)


@factory
class Ty(biclosed.Ty):
    "Base class for categorial grammar types."
    variable_factory: ClassVar[type["Variable"]]
    constant_factory: ClassVar[type["Constant"]]
    abstraction_factory: ClassVar[type["Abstraction"]]
    over_factory: ClassVar[type["Over"]]
    under_factory: ClassVar[type["Under"]]


class Over(biclosed.Over):
    "Categorial grammar type ``base << exponent``."

    ob = Ty


class Under(biclosed.Under):
    "Categorial grammar type ``exponent >> base``."

    ob = Ty


@factory
class Diagram(biclosed.Diagram):
    """
    A categorial diagram is a biclosed diagram with rules and words as boxes.
    """
    box_factory: ClassVar[type["Box"]]
    functor_factory: ClassVar[type["Functor"]]
    curry_factory: ClassVar[type["Curry"]]
    eval_factory: ClassVar[type["Eval"]]
    coeval_factory: ClassVar[type["Coeval"]]
    ob = Ty

    def to_pregroup(self):
        from discopy.lambek import pregroup

        return Functor(
            ob_map=lambda x: pregroup.Ty(x.inside[0].name),
            ar_map=lambda f: pregroup.Box(f.name,
                                          Diagram.to_pregroup(f.dom),
                                          Diagram.to_pregroup(f.cod)),
            cod=pregroup.Diagram)(self)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        return Eval(left << right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return Eval(left >> right)

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return (
            (left << middle) @ Eval(middle << right) >> Eval(left << middle)
        ).curry(n=len(right), left=True)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return (
            Eval(left >> middle) @ (middle >> right) >> Eval(middle >> right)
        ).curry(n=len(left), left=False)

    @staticmethod
    def fx(left, middle, right):
        """ Forward crossed composition. """
        return ForwardCrossedComposition(left << middle, right >> middle)

    @staticmethod
    def bx(left, middle, right):
        """ Backward crossed composition. """
        return BackwardCrossedComposition(middle << left, middle >> right)


class Box(biclosed.Box, Diagram):
    """
    A categorial box is a grammar rule in a categorial diagram.
    """


class Word(thue.Word, Box):
    """
    A categorial word is a rule with a ``name``, a grammatical type as ``cod``
    and an optional domain ``dom``.

    Parameters:
        name (str) : The name of the word.
        cod (biclosed.Ty) : The grammatical type of the word.
        dom (biclosed.Ty) : An optional domain for the word, empty by default.
    """


class Eval(biclosed.Eval, Box):
    """
    Evaluation box in a categorial grammar.
    """


class Coeval(biclosed.Coeval, Box):
    """
    Coevaluation box in a categorial grammar, the dagger of an evaluation.
    """


class Curry(biclosed.Curry, Box):
    """
    The currying of a categorial diagram.
    """


class Bubble(biclosed.Bubble, Box):
    """
    A categorial bubble is a biclosed bubble and a categorial box.

    >>> N = Ty('N')
    >>> assert (Box('f', N, N).bubble() >> Box('g', N, N)).cod == N
    """


class ForwardCrossedComposition(BinaryBoxConstructor, Box):
    """ Forward crossed composition rule. """
    serialised_attrs = ("left", "right")

    def __init__(self, left, right):
        assert left.is_over
        assert right.is_under
        if left.exponent != right.base:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.exponent, right.base))
        name = f"ForwardCrossedComposition({left}, {right})"
        dom, cod = left @ right, right.exponent >> left.base
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class BackwardCrossedComposition(BinaryBoxConstructor, Box):
    """ Backward crossed composition rule. """
    serialised_attrs = ("left", "right")

    def __init__(self, left, right):
        assert left.is_over
        assert right.is_under
        if left.base != right.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.base, right.exponent))
        name = f"BackwardCrossedComposition({left}, {right})"
        dom, cod = left @ right, right.base << left.exponent
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class Functor(biclosed.Functor):
    """
    A categorial functor is a biclosed functor with a predefined mapping
    for categorial rules.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, ForwardCrossedComposition):
            left = other.dom.inside[0].left
            middle = other.dom.inside[0].right
            right = other.dom.inside[1].left
            return self.cod.fx(self(left), self(middle), self(right))
        if isinstance(other, BackwardCrossedComposition):
            left = other.dom.inside[0].right
            middle = other.dom.inside[0].left
            right = other.dom.inside[1].right
            return self.cod.bx(self(left), self(middle), self(right))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a categorial grammar.
    """
    functor = Functor.id(Diagram)
    freevars = None

    def __call__(self, other, left=False):
        return BA(self, other) if left else FA(self, other)

    def to_abstract(self):
        """
        The abstract term of a categorial term, dropping planarity,
        see :meth:`discopy.lambek.abstract.Diagram.from_categorial`.
        """
        from discopy.lambek.abstract import Diagram
        return Diagram.from_categorial(self)


class Constant(TermBase, biclosed.Constant):
    serialised_attrs = ("name", "cod")

    def __init__(self, name: str, cod: Ty):
        biclosed.Constant.__init__(self, name, cod)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return self


class Variable(TermBase, biclosed.Variable):
    serialised_attrs = biclosed.Variable.serialised_attrs

    def simplify(self):
        return self


class Abstraction(TermBase, biclosed.Abstraction):
    serialised_attrs = biclosed.Abstraction.serialised_attrs
    var: Variable
    body: Term
    left: bool = False

    def __init__(self, var: Variable, body: Term, left: bool = False):
        biclosed.Abstraction.__init__(self, var, body, left)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return Abstraction(self.var, self.body.simplify(), self.left)


class FA(TermBase, biclosed.Application):
    "Application of type ``Y`` with subterms of type ``Y << X`` and ``X``."
    serialised_attrs = ("func", "args")

    def __init__(self, func, args):
        biclosed.Application.__init__(self, func, args, left=False)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['func'], tree['args'])))

    def simplify(self):
        return self.func.simplify()(self.args.simplify())


class BA(TermBase, biclosed.Application):
    "Application of type ``Y`` with subterms of type ``X`` and ``X >> Y``."
    serialised_attrs = ("args", "func")

    def __init__(self, args, func):
        biclosed.Application.__init__(self, func, args, left=True)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['args'], tree['func'])))

    def simplify(self):
        return self.args.simplify()(self.func.simplify(), left=True)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.args!r}, {self.func!r})"


class TypeRaising(TermBase):
    "Abstract superclass of :class:`FTR` and :class:`BTR`."
    serialised_attrs = ("base", "child", "cod")
    base: Ty
    child: Term

    def __init__(self, base, child, cod):
        name = f"{type(self).__name__}({base}, {child})"
        self.base, self.child, self.freevars = base, child, child.freevars
        super().__init__(name, child.dom, cod)

    def eval(self, functor=None):
        return self.simplify().eval(functor)

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Type raisings of one base with alpha-equivalent children. """
        if any(type(other) is not type(self) or other.base != self.base
               for other in others):
            return False
        return self.child.alpha_eq_under(
            scopes, [other.child for other in others], depth)

    def map(self, functor, context):
        return functor.map_term(self.simplify(), context)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.base!r}, {self.child!r})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)), base=self.base.to_tree(),
                    child=self.child.to_tree())

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['base'], tree['child'])))

    @property
    def constants(self):
        return self.child.constants


class FTR(TypeRaising):
    "Forward type raising ``Y << (X >> Y)`` with base ``Y`` and child ``X``."
    serialised_attrs = ("base", "child")

    def __init__(self, base, child):
        super().__init__(base, child, base << (child.cod >> base))

    def simplify(self):
        child = self.child.simplify()
        f = Variable.fresh(
            "f", self.child.cod >> self.base, child)
        return Abstraction(f, child(f, left=True))


class BTR(TypeRaising):
    "Backward type raising ``(Y << X) >> Y`` with base ``Y`` and child ``X``."
    serialised_attrs = ("base", "child")

    def __init__(self, base, child):
        super().__init__(base, child, (base << child.cod) >> base)

    def simplify(self):
        child = self.child.simplify()
        f = Variable.fresh(
            "f", self.base << self.child.cod, child)
        return Abstraction(f, f(child), left=True)


class BinaryTerm(TermBase):
    "Abstract superclass of :class:`FC`, :class:`BC`, :class:`FX`, :class:`BX`"
    serialised_attrs = ("left", "right")
    left: Term
    right: Term

    def __init__(self, left, right, cod):
        if set(left.freevars).intersection(right.freevars):
            raise ValueError("Expected disjoint free variables.")
        name = f"{type(self).__name__}({left}, {right})"
        self.left, self.right = left, right
        self.freevars = left.freevars + right.freevars
        super().__init__(name, left.dom @ right.dom, cod)

    def simplify(self):
        return type(self)(  # ty: ignore[missing-argument]
            self.left.simplify(), self.right.simplify())

    def eval(self, functor=None):
        return self.simplify().eval(functor)

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Compositions of one kind with alpha-equivalent sides. """
        if any(type(other) is not type(self) for other in others):
            return False
        return self.left.alpha_eq_under(
            scopes, [other.left for other in others], depth)\
            and self.right.alpha_eq_under(
                scopes, [other.right for other in others], depth)

    def map(self, functor, context):
        return functor.map_term(self.simplify(), context)

    def setoid(self):
        """
        Binary terms are equal when their kinds and sides are, see
        :meth:`discopy.biclosed.Application.setoid`.
        """
        return factory_key(type(self)), self.left, self.right

    def __repr__(self):
        return factory_name(type(self)) + f"({self.left!r}, {self.right!r})"

    def to_tree(self):
        return dict(factory=factory_name(type(self)), left=self.left.to_tree(),
                    right=self.right.to_tree())

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))

    @property
    def constants(self):
        return self.left.constants + self.right.constants


class FC(BinaryTerm):
    "Forward composition ``A << C`` with subterms ``A << B`` and ``B << C``. "
    def __init__(self, left, right):
        assert left.cod.is_over
        assert right.cod.is_over
        if right.cod.base != left.cod.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left.cod, right.cod, left.cod.exponent, right.cod.base))
        super().__init__(left, right, left.cod.base << right.cod.exponent)

    def simplify(self):
        f, g = self.left.simplify(), self.right.simplify()
        x = Variable.fresh(
            "x", self.right.cod.exponent, f, g)
        return Abstraction(x, f(g(x)))


class BC(BinaryTerm):
    "Backward composition ``A >> C`` with subterms ``A >> B`` and ``B >> C``."
    def __init__(self, left, right):
        assert left.cod.is_under
        assert right.cod.is_under
        if left.cod.base != right.cod.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left.cod, right.cod, left.cod.base, right.cod.exponent))
        super().__init__(left, right, left.cod.exponent >> right.cod.base)

    def simplify(self):
        f, g = self.left.simplify(), self.right.simplify()
        x = Variable.fresh(
            "x", self.left.cod.exponent, f, g)
        return Abstraction(x, x(f, left=True)(g, left=True), left=True)


class FX(BinaryTerm):
    "Forward crossing ``A >> C`` with subterms ``C << B`` and ``A >> B``."
    def __init__(self, left, right):
        rule = ForwardCrossedComposition(left.cod, right.cod)
        super().__init__(left, right, rule.cod)

    def eval(self, functor=None):
        functor, X = functor or self.functor, self.left.cod.base
        Y, Z = self.left.cod.exponent, self.right.cod.exponent
        f, g = self.left.eval(functor), self.right.eval(functor)
        return f @ g >> functor.cod.fx(*map(functor, [X, Y, Z]))

    def map(self, functor, context):
        if not issubclass(get_origin(functor.cod), ClosedCategory):
            raise AxiomError(
                "Crossed composition requires a closed codomain.")
        ob = functor.cod.ob
        f, g = (functor.map_term(term, context)
                for term in (self.left, self.right))
        var = ob.variable_factory.fresh(
            "x", functor(self.right.cod.exponent), f, g)
        return ob.abstraction_factory(var, f(g(var)))


class BX(BinaryTerm):
    "Backward crossing ``A << C`` with subterms ``B << C`` and ``B >> A``."
    def __init__(self, left, right):
        rule = BackwardCrossedComposition(left.cod, right.cod)
        super().__init__(left, right, rule.cod)

    def eval(self, functor=None):
        functor, Z = functor or self.functor, self.right.cod.base
        X, Y = self.left.cod.exponent, self.left.cod.base
        f, g = self.left.eval(functor), self.right.eval(functor)
        return f @ g >> functor.cod.bx(*map(functor, [X, Y, Z]))

    def map(self, functor, context):
        if not issubclass(get_origin(functor.cod), ClosedCategory):
            raise AxiomError(
                "Crossed composition requires a closed codomain.")
        ob = functor.cod.ob
        f, g = (functor.map_term(term, context)
                for term in (self.left, self.right))
        var = ob.variable_factory.fresh(
            "x", functor(self.left.cod.exponent), f, g)
        return ob.abstraction_factory(var, g(f(var)))


type Term = (
    Constant | Variable | Abstraction | FTR | BTR
    | FA | BA | FC | BC | FX | BX)


def cat2ty(string: str) -> Ty:
    r"""
    Translate the string representation of a CCG category into DisCoPy.

    A category is either an atom, a bracketed category or two categories
    around a slash, which associates to the left as in Steedman's
    `The Syntactic Process (2000) <https://mitpress.mit.edu/9780262527446>`_.
    The features that CCGbank writes in square brackets, e.g. the ``[dcl]`` of
    ``S[dcl]``, are ignored: they refine an atom rather than change it.

    Parameters:
        string : The string with slashes representing a CCG category.

    Example
    -------
    >>> print(cat2ty(r"(S[dcl]\NP)/NP"))
    ((NP >> S) << NP)
    >>> assert cat2ty(r"S\NP/NP") == cat2ty(r"(S\NP)/NP")
    >>> assert cat2ty(r"(S\NP)\(S\NP)[conj]") == cat2ty(r"(S\NP)\(S\NP)")
    """
    def split(string):
        par_count = 0
        for i, char in reversed(list(enumerate(string))):
            if char == ")":
                par_count += 1
            elif char == "(":
                par_count -= 1
            elif char in ["\\", "/"] and par_count == 0:
                return string[:i], char, string[i + 1:]
        return string, None, None

    string = re.sub(r'\[[^]]*\]', '', string)
    left, slash, right = split(string)
    if slash == '\\':
        return cat2ty(right) >> cat2ty(left)
    if slash == '/':
        return cat2ty(left) << cat2ty(right)
    if string.startswith('(') and string.endswith(')'):
        return cat2ty(string[1:-1])
    return Ty(string)


def tree2diagram(tree: dict, dom=Ty()) -> Diagram:
    """
    Translate a depccg.Tree in JSON format into DisCoPy.

    Parameters:
        tree : The tree to translate.
        dom : The domain for the word boxes, empty by default.
    """
    if 'word' in tree:
        return Word(tree['word'], cat2ty(tree['cat']), dom=dom)
    children = [tree2diagram(child, dom) for child in tree['children']]
    dom = Ty().tensor(*[child.cod for child in children])
    cod = cat2ty(tree['cat'])
    if tree['type'] == 'ba':
        rule = Diagram.ba(dom[:1], dom.inside[1].base)
    elif tree['type'] == 'fa':
        rule = Diagram.fa(dom.inside[0].base, dom[1:])
    elif tree['type'] == 'fc':
        rule = Diagram.fc(
            dom.inside[0].base,
            dom.inside[0].exponent,
            dom.inside[1].exponent)
    else:
        rule = Box(tree['type'], dom, cod)
    return Id().tensor(*children) >> rule


def combine(left: Term, right: Term) -> list[Term]:
    """
    The terms of two adjacent terms by forward and backward application and
    composition, in Eisner's normal form: a composition is not the functor
    of the next rule in its direction (Eisner, `Efficient normal-form
    parsing for combinatory categorial grammar (1996)
    <https://aclanthology.org/P96-1011/>`_), so that the terms of one
    meaning are combined once.

    Parameters:
        left : The term on the left.
        right : The term on the right.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> f, g, a = (x << y)("f"), (y << z)("g"), z("a")
    >>> print(*combine(f, g))
    FC((x << y)('f'), (y << z)('g'))
    >>> assert combine(combine(f, g)[0], a) == []
    """
    x, y, result = left.cod, right.cod, []
    if x.is_over and not isinstance(left, FC):
        if x.exponent == y:
            result.append(FA(left, right))
        if y.is_over and x.exponent == y.base:
            result.append(FC(left, right))
    if y.is_under and not isinstance(right, BC):
        if y.exponent == x:
            result.append(BA(left, right))
        if x.is_under and x.base == y.exponent:
            result.append(BC(left, right))
    return result


def parse(sentence, k: int = 1, goal: Ty | None = None
          ) -> list[tuple[Term, float]]:
    """
    The ``k`` best terms of a sentence with their score, the sum of the
    scores of their words, by the CKY algorithm over the rules of
    :func:`combine`: the chart keeps the ``k`` best terms of each type
    for each span of words (Lewis and Steedman, `A* CCG parsing with a
    supertag-factored model (2014) <https://aclanthology.org/D14-1107/>`_).

    Parameters:
        sentence : The candidate constants of each word with their score,
            e.g. the log-probabilities of a supertagger.
        k : The number of terms.
        goal : The type of the terms, any type by default.

    Example
    -------
    >>> n, s = Ty("n"), Ty("s")
    >>> Alice, Bob = n("Alice"), n("Bob")
    >>> loves, lovely = ((n >> s) << n)("loves"), (n << n)("loves")
    >>> sentence = [[(Alice, 0.)], [(lovely, -.5), (loves, -1.)], [(Bob, 0.)]]
    >>> for term, score in parse(sentence, k=2):
    ...     print(term.cod, score)
    s -1.0
    >>> (term, _), = parse(sentence, goal=s)
    >>> assert term == Alice(loves(Bob), left=True)
    """
    def best(items):
        return sorted(items, key=lambda item: -item[1])[:k]

    def cell(items):
        result = {}
        for term, score in items:
            result.setdefault(term.cod, []).append((term, score))
        return {cod: best(terms) for cod, terms in result.items()}

    chart = {(i, i + 1): cell(candidates)
             for i, candidates in enumerate(sentence)}
    for width in range(2, len(sentence) + 1):
        for i in range(len(sentence) - width + 1):
            chart[i, i + width] = cell(
                (term, a + b)
                for middle in range(i + 1, i + width)
                for lefts in chart[i, middle].values()
                for left, a in lefts
                for rights in chart[middle, i + width].values()
                for right, b in rights
                for term in combine(left, right))
    return best(
        item for cod, items in chart.get((0, len(sentence)), {}).items()
        if goal is None or cod == goal for item in items)


PUNCTUATION = frozenset({
    ",", ".", ":", ";", "LRB", "RRB", "-LRB-", "-RRB-", "``", "''"})
EMPTY_SPAN = ":"
PR = re.compile(r"(?<![A-Za-z0-9_\]])PR(?![A-Za-z0-9_])")


def normalise(label: str) -> str:
    """
    The category of a label as Rebank writes it: ``NP/N`` is ``NP[nb]/N``
    and the atom ``PR`` is ``PP``.

    >>> print(normalise("NP/N"), normalise(r"(S\\NP)/PR"))
    NP[nb]/N (S\\NP)/PP
    """
    return PR.sub("PP", {"NP/N": "NP[nb]/N"}.get(label, label))


@cache
def structure(category: str) -> str | tuple | None:
    r"""
    A category as an atom or a triple ``(slash, result, argument)``, its
    slashes associating to the left, ``None`` if it is malformed. Unlike
    :func:`cat2ty`, it keeps the features.

    >>> structure(r"(S[dcl]\NP)/NP")
    ('/', ('\\', 'S[dcl]', 'NP'), 'NP')
    """
    category = category.strip()
    while category.startswith("(") and category.endswith(")"):
        depth = 0
        for i, char in enumerate(category):
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth == 0 and i != len(category) - 1:
                break
        else:
            category = category[1:-1].strip()
            continue
        break
    depth = 0
    for i in reversed(range(len(category))):
        depth += {")": 1, "(": -1}.get(category[i], 0)
        if depth == 0 and category[i] in "/\\":
            result = structure(category[:i])
            argument = structure(category[i + 1:])
            if result is None or argument is None:
                return None
            return category[i], result, argument
    return category or None


def render(category: str | tuple) -> str:
    """ The string of a category split by :func:`structure`. """
    if isinstance(category, str):
        return category
    slash, *sides = category
    result, argument = (
        render(x) if isinstance(x, str) else f"({render(x)})" for x in sides)
    return result + slash + argument


@cache
def schemata(  # pylint: disable=too-many-locals
        left: str, right: str) -> frozenset[str]:
    r"""
    The categories that the schemata of CCG derive from two adjacent ones:
    application, harmonic composition, a modifier ``X|X`` applied to
    anything, punctuation, conjunction and coordination.

    >>> sorted(schemata(r"NP", r"S[dcl]\NP"))
    ['S[dcl]']
    >>> sorted(schemata(r"S/S", r"S/NP"))
    ['S/NP']
    """
    parents = set()
    left_base, left_conj = (left[:-6], True) if left.endswith("[conj]")\
        else (left, False)
    right_base, right_conj = (right[:-6], True) if right.endswith("[conj]")\
        else (right, False)
    x, y = structure(left_base), structure(right_base)
    if x is None or y is None:
        return frozenset()
    for this, other in ((left, right_base), (right, left_base)):
        if this == "conj":
            parents.add(f"{other}[conj]")
        if this in PUNCTUATION:
            parents.add(other)
    if (left_conj or right_conj) and left_base == right_base:
        parents.add(left_base)
    (x_slash, *x_sides), (y_slash, *y_sides) = (
        c if isinstance(c, tuple) else ("", c, c) for c in (x, y))
    if x_slash == "/" and x_sides[1] == y:
        parents.add(render(x_sides[0]))
    if y_slash == "\\" and y_sides[1] == x:
        parents.add(render(y_sides[0]))
    x_modifies, y_modifies = (
        bool(slash) and sides[0] == sides[1]
        for slash, sides in ((x_slash, x_sides), (y_slash, y_sides)))
    if x_modifies:
        parents.add(right_base)
    if y_modifies:
        parents.add(left_base)
    if x_modifies and y_modifies and x == y:
        parents.add(render(x_sides[0]))
    if x_slash == y_slash == "/" and x_sides[1] == y_sides[0]:
        parents.add(render(("/", x_sides[0], y_sides[1])))
    if x_slash == y_slash == "\\" and y_sides[1] == x_sides[0]:
        parents.add(render(("\\", y_sides[0], x_sides[1])))
    return frozenset(parents)


@dataclass(frozen=True)
class Derivation:
    """
    A derivation of a span of words in the chart of a :class:`Parser`.

    Parameters:
        cat : The category it derives, as a string.
        score : Its score, a sum of log-probabilities and penalties.
        word : The word of a lexical derivation.
        children : The one or two derivations it combines.
    """
    cat: str
    score: float = 0.
    word: str | None = None
    children: tuple[Derivation, ...] = ()

    @property
    def rule(self) -> str:
        """
        The rule of the derivation, read from the types of :func:`cat2ty`:
        ``"fa"``, ``"ba"`` or ``"fc"`` when it is one, ``"lex"`` for a word,
        ``"u"`` for any other unary rule and ``"rule"`` for any other.
        """
        if self.word is not None:
            return "lex"
        if len(self.children) == 1:
            return "u"
        x, y, z = (cat2ty(c.cat) for c in self.children + (self, ))
        if x.is_over and x.exponent == y and x.base == z:
            return "fa"
        if y.is_under and y.exponent == x and y.base == z:
            return "ba"
        if x.is_over and y.is_over and x.exponent == y.base\
                and z == x.base << y.exponent:
            return "fc"
        return "rule"

    def to_tree(self) -> dict:
        """ The derivation as a tree in the JSON format of depccg. """
        if self.word is not None:
            return {"word": self.word, "cat": self.cat}
        return {"type": self.rule, "cat": self.cat,
                "children": [child.to_tree() for child in self.children]}

    def to_diagram(self) -> Diagram:
        """
        The derivation as a categorial diagram, see :func:`tree2diagram`.

        >>> alice, runs = Derivation("NP", word="Alice"), Derivation(
        ...     r"S[dcl]\\NP", word="runs")
        >>> print(Derivation("S[dcl]", children=(alice, runs)).to_diagram())
        Alice >> NP @ runs >> Eval(NP >> S)
        """
        return tree2diagram(self.to_tree())


class Parser:  # pylint: disable=too-many-instance-attributes
    r"""
    The chart parser of the reproduction of Bobcat (Clark, `Something old,
    something new: grammar-based CCG parsing with transformer models (2021)
    <https://arxiv.org/abs/2109.10044>`_) trained on CCGrebank, whose model
    is `rel-int/bobcat-rebank-deberta-v3-large
    <https://huggingface.co/rel-int/bobcat-rebank-deberta-v3-large>`_ on
    Hugging Face: a CKY chart over the binary and unary rules observed in
    the treebank, scored by a supertagger for the words and a span classifier
    for the constituents.

    A derivation scores the sum of the log-probabilities of its supertags
    and of the best label of each of its spans that has its category, less a
    penalty ``generic`` for a binary rule licensed by the schemata of
    :func:`schemata` only, and ``missing`` for a span with no such label. A
    cell keeps the best derivation of each category, then the ``beam`` best,
    and a sentence is parsed when its last cell has a ``root`` category.
    Ties go to the first derivation found, the parents of a rule taken in
    the order of their strings.

    Parameters:
        binary : The parents of each pair of categories in the treebank.
        unary : The parents of each category in the treebank.
        root : The categories of a sentence.
        beam : The size of each cell.
        generic : The penalty of a generic rule, ``None`` for none.
        missing : The penalty of a missing span, ``None`` for none.

    Example
    -------
    >>> parser = Parser.from_json({
    ...     "binary": [[r"(S[dcl]\NP)/NP", "NP", r"S[dcl]\NP"]],
    ...     "unary": [["N", "NP"]], "root_categories": ["S[dcl]"]})
    >>> tags = [[("N", -.1)], [(r"(S[dcl]\NP)/NP", -.2)], [("NP", -.3)]]
    >>> spans = {(1, 3): [(r"S[dcl]\NP", -.4)], (0, 3): [("S[dcl]", -.5)]}
    >>> tree = parser.parse("Alice loves Bob".split(), tags, spans)
    >>> print(tree.cat, tree.rule, round(tree.score, 2))
    S[dcl] ba -5.55

    The score is the sum of the five log-probabilities, less the penalty 4
    of the span of ``Alice`` as ``NP``, which has no label, and the penalty
    .05 of backward application, which the treebank has not seen here.
    >>> print(tree.to_diagram().cod)
    S
    """
    def __init__(  # pylint: disable=too-many-arguments
            self, binary: dict[tuple[str, str], frozenset[str]],
            unary: dict[str, frozenset[str]], *,
            root: frozenset[str] | None = None,
            beam: int = 256, generic: float | None = .05,
            missing: float | None = 4.):
        self.binary, self.unary, self.root = binary, unary, root
        self.beam, self.generic, self.missing = beam, generic, missing
        self.encoder = self.classifier = None
        self.decoding: dict[str, float] = {}

    @classmethod
    def from_json(cls, data: dict, **params) -> Parser:
        """
        A parser from the ``native_grammar.json`` of a model.

        Parameters:
            data : Its ``binary`` triples, ``unary`` pairs and
                ``root_categories``.
            params : The other parameters of :class:`Parser`.
        """
        binary, unary = {}, {}
        for left, right, parent in data.get("binary", []):
            binary.setdefault((left, right), set()).add(parent)
        for child, parent in data.get("unary", []):
            unary.setdefault(child, set()).add(parent)
        params.setdefault(
            "root", frozenset(data.get("root_categories", [])) or None)
        return cls({key: frozenset(value) for key, value in binary.items()},
                   {key: frozenset(value) for key, value in unary.items()},
                   **params)

    @classmethod
    def from_pretrained(  # pylint: disable=too-many-locals
            cls, directory, device: str = "cpu", **params) -> Parser:
        """
        A parser from the directory of a model, e.g. a download of
        `rel-int/bobcat-rebank-deberta-v3-large
        <https://huggingface.co/rel-int/bobcat-rebank-deberta-v3-large>`_:
        its ``native_grammar.json``, ``label_maps.json``,
        ``rebank_pipeline_config.json``, the configuration and the tokenizer
        of its encoder and its weights ``pytorch_model.bin``. It needs
        ``transformers``.

        Parameters:
            directory : The directory, a string or a path.
            device : Where the model runs.
            params : The parameters of :class:`Parser`, by default those of
                the model's pipeline.
        """
        # pylint: disable=import-outside-toplevel  # torch is an extra
        import json
        from pathlib import Path

        import torch
        from transformers import (  # pylint: disable=import-error
            AutoConfig, AutoModel, AutoTokenizer)

        from discopy.lambek.neural import ChartClassifier, Encoder

        path = Path(directory)
        pipeline = json.loads(
            (path / "rebank_pipeline_config.json").read_text())
        maps = json.loads((path / "label_maps.json").read_text())
        params.setdefault("beam", pipeline["parser"]["beam_size"])
        params.setdefault(
            "generic", pipeline["parser"]["generic_rule_penalty"]
            if pipeline["parser"]["use_generic_rules"] else None)
        params.setdefault(
            "missing", pipeline["parser"]["missing_span_penalty"])
        parser = cls.from_json(json.loads(
            (path / "native_grammar.json").read_text()), **params)
        config = AutoConfig.from_pretrained(path)
        parser.encoder = Encoder(
            AutoModel.from_config(config), AutoTokenizer.from_pretrained(path),
            config.max_position_embeddings)
        tags, labels = (sorted(maps[key], key=maps[key].get)
                        for key in ("tag2id", "cat2id"))
        parser.classifier = ChartClassifier(tags, labels, config.hidden_size)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        parser.encoder.model.load_state_dict({
            key[len("bert."):]: value for key, value in state.items()
            if key.startswith("bert.")})
        parser.classifier.load_state_dict({
            key: value for key, value in state.items()
            if not key.startswith("bert.")})
        parser.encoder.to(device).eval()
        parser.classifier.to(device).eval()
        parser.decoding = {
            "tag_k": pipeline["decoder"]["tag_top_k"],
            "tag_threshold": pipeline["decoder"]["tag_prob_threshold"],
            "span_k": pipeline["decoder"]["span_top_k"],
            "span_threshold": pipeline["decoder"]["span_prob_threshold"]}
        return parser

    def decode(self, sentences: list[list[str]]) -> list[tuple[list, dict]]:
        """
        The supertags and the labels of the spans of each sentence, by the
        model of :meth:`from_pretrained`, see
        :meth:`discopy.lambek.neural.ChartClassifier.decode`.
        """
        if self.encoder is None or self.classifier is None:
            raise ValueError("no model: see Parser.from_pretrained")
        states, mask = self.encoder(sentences)
        return self.classifier.decode(states, mask, **self.decoding)

    def __call__(self, sentence: str | list[str]) -> Diagram | None:
        """
        The diagram of a sentence, its words split on spaces if it is a
        string, ``None`` if it has no parse: the best parse with a label for
        every span, else the best parse, as Bobcat's command line does.
        """
        tree = self.derive(sentence)
        return None if tree is None else tree.to_diagram()

    def derive(self, sentence: str | list[str]) -> Derivation | None:
        """ The derivation of :meth:`__call__`. """
        words = sentence.split() if isinstance(sentence, str) else sentence
        if not words:
            return None
        tags, spans = self.decode([words])[0]
        words = words[:len(tags)]
        return self.parse(words, tags, spans, missing=False)\
            or self.parse(words, tags, spans)

    def parents(self, left: str, right: str) -> list[str]:
        """ The parents of two categories, observed or generic. """
        parents = set(self.binary.get((left, right), ()))
        if self.generic is not None:
            parents |= schemata(left, right)
        return sorted(parents)

    def span_score(self, cat: str, labels: dict[str, float] | None,
                   missing: bool) -> float | None:
        """
        The log-probability of a category in a span, see :meth:`labels`, the
        penalty of a missing span if it has none and ``missing``, else
        ``None``.
        """
        if labels is None:
            return 0.
        if cat in labels:
            return labels[cat]
        return -self.missing if missing and self.missing is not None\
            else None

    @staticmethod
    def labels(labels: list[tuple[str, float]]) -> dict[str, float]:
        """
        The best log-probability of a label of a span with each category,
        that of a chain of unary rules ``"N::S[b]\\NP"`` counting for each.
        """
        best = {}
        for label, score in labels:
            for part in label.split("::"):
                if part and part != EMPTY_SPAN:
                    part = normalise(part)
                    best[part] = max(best.get(part, score), score)
        return best

    @staticmethod
    def add(cell: dict[str, Derivation], tree: Derivation) -> bool:
        """ Keep a derivation in a cell if it beats that of its category. """
        if tree.cat in cell and cell[tree.cat].score >= tree.score:
            return False
        cell[tree.cat] = tree
        return True

    def close(self, cell: dict[str, Derivation], labels, missing: bool):
        """ Add to a cell every derivation its unary rules reach. """
        queue = list(cell.values())
        while queue:
            child = queue.pop(0)
            for parent in sorted(self.unary.get(child.cat, ())):
                score = self.span_score(parent, labels, missing)
                if score is not None and self.add(cell, Derivation(
                        parent, child.score + score, children=(child, ))):
                    queue.append(cell[parent])

    def prune(self, cell: dict[str, Derivation]) -> dict[str, Derivation]:
        """ The ``beam`` best derivations of a cell. """
        if self.beam <= 0 or len(cell) <= self.beam:
            return cell
        return dict(sorted(
            cell.items(), key=lambda item: -item[1].score)[:self.beam])

    def chart(self, words: list[str], tags: list[list[tuple[str, float]]],
              spans: dict[tuple[int, int], list[tuple[str, float]]] | None,
              missing: bool = True) -> dict[tuple[int, int], dict]:
        """
        The chart of a sentence, a cell for each span of words from its first
        to the one after its last, each cell the best derivation of each
        category.

        Parameters:
            words : The words of the sentence.
            tags : The supertags of each word with their log-probability.
            spans : The labels of each span with their log-probability,
                ``None`` for no span classifier.
            missing : Whether a span may have a category it has no label of.
        """
        if len(words) != len(tags):
            raise ValueError(
                f"{len(words)} words but {len(tags)} lists of tags")
        chart = {}
        best = {} if spans is None else {
            span: self.labels(value) for span, value in spans.items()}

        def labels(span):
            return None if spans is None else best.get(span, {})
        for i, (word, candidates) in enumerate(zip(words, tags)):
            chart[i, i + 1] = self.lexical(
                word, candidates, labels((i, i + 1)), missing)
        for width in range(2, len(words) + 1):
            for start in range(len(words) - width + 1):
                self.fill(chart, start, start + width,
                          labels=labels((start, start + width)),
                          missing=missing)
        return chart

    def fill(  # pylint: disable=too-many-arguments
            self, chart, start: int, end: int, *, labels, missing: bool):
        """ Fill the cell of a span from every split of it in two. """
        cell = {}
        for middle in range(start + 1, end):
            for left in chart.get((start, middle), {}).values():
                for right in chart.get((middle, end), {}).values():
                    self.combine(cell, left, right, labels=labels,
                                 missing=missing)
        if cell:
            self.close(cell, labels, missing)
            chart[start, end] = self.prune(cell)

    def lexical(self, word: str, candidates: list[tuple[str, float]],
                labels, missing: bool) -> dict[str, Derivation]:
        """ The cell of a word, its supertags closed under unary rules. """
        cell = {}
        for tag, score in candidates:
            self.add(cell, Derivation(normalise(tag), score, word))
        self.close(cell, labels, missing)
        return self.prune(cell)

    def combine(  # pylint: disable=too-many-arguments
            self, cell, left: Derivation, right: Derivation, *, labels,
            missing: bool):
        """ Add to a cell the derivations of two adjacent ones. """
        seen = self.binary.get((left.cat, right.cat), ())
        for parent in self.parents(left.cat, right.cat):
            score = self.span_score(parent, labels, missing)
            if score is None:
                continue
            penalty = 0. if parent in seen or self.generic is None\
                else self.generic
            self.add(cell, Derivation(
                parent, left.score + right.score + score - penalty,
                children=(left, right)))

    def parse(self, words, tags, spans=None, missing: bool = True
              ) -> Derivation | None:
        """
        The best derivation of a sentence with a root category, ``None`` if
        there is none, see :meth:`chart`.
        """
        if not words:
            return None
        cell = self.chart(words, tags, spans, missing).get((0, len(words)), {})
        candidates = [tree for tree in cell.values()
                      if self.root is None or tree.cat in self.root]
        return max(candidates, key=lambda tree: tree.score, default=None)


Id = Diagram.id
Diagram.box_factory, Diagram.functor_factory = Box, Functor
Diagram.curry_factory = Curry
Diagram.bubble_factory = Bubble
Diagram.eval_factory, Diagram.coeval_factory = Eval, Coeval

Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory =\
    lambda func, args, left=False: BA(args, func) if left else FA(func, args)
Ty.abstraction_factory = Abstraction

Ty.over_factory, Ty.under_factory = Over, Under
