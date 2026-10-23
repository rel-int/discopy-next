# -*- coding: utf-8 -*-

"""
A categorial grammar is a free biclosed category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    TermBase
    Constant
    Variable
    Abstraction
    FA
    BA
    FC
    BC
    FX
    BX
    Diagram
    Box
    Word
    ForwardCrossedComposition
    BackwardCrossedComposition
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
"""

from __future__ import annotations

import re
from typing import ClassVar

from discopy import biclosed, cmap, messages
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.grammar import thue
from discopy.utils import (
    BinaryBoxConstructor,
    AxiomError,
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
        from discopy.grammar import pregroup

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


class ForwardCrossedComposition(BinaryBoxConstructor, Box):
    """ Forward crossed composition rule. """
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
        see :meth:`discopy.grammar.abstract.Diagram.from_categorial`.
        """
        from discopy.grammar.abstract import Diagram
        return Diagram.from_categorial(self)


class Constant(TermBase, biclosed.Constant):
    def __init__(self, name: str, cod: Ty):
        biclosed.Constant.__init__(self, name, cod)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return self


class Variable(TermBase, biclosed.Variable):
    def simplify(self):
        return self


class Abstraction(TermBase, biclosed.Abstraction):
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
    def __init__(self, base, child):
        super().__init__(base, child, base << (child.cod >> base))

    def simplify(self):
        child = self.child.simplify()
        f = Variable.fresh(
            "f", self.child.cod >> self.base, child)
        return Abstraction(f, child(f, left=True))


class BTR(TypeRaising):
    "Backward type raising ``(Y << X) >> Y`` with base ``Y`` and child ``X``."
    def __init__(self, base, child):
        super().__init__(base, child, (base << child.cod) >> base)

    def simplify(self):
        child = self.child.simplify()
        f = Variable.fresh(
            "f", self.base << self.child.cod, child)
        return Abstraction(f, f(child), left=True)


class BinaryTerm(TermBase):
    "Abstract superclass of :class:`FC`, :class:`BC`, :class:`FX`, :class:`BX`"
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
        return type(self)(self.left.simplify(), self.right.simplify())

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
        return type(self), self.left, self.right

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
    "Forward crossing ``A >> C`` with subterms ``B << A`` and ``B >> C``."
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
    "Backward crossing ``A << C`` with subterms ``A << B`` and ``C >> B``."
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
    children = list(map(tree2diagram, tree['children']))
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


Id = Diagram.id
Diagram.box_factory, Diagram.functor_factory = Box, Functor
Diagram.curry_factory = Curry
Diagram.eval_factory, Diagram.coeval_factory = Eval, Coeval

Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory =\
    lambda func, args, left=False: BA(args, func) if left else FA(func, args)
Ty.abstraction_factory = Abstraction

Ty.over_factory, Ty.under_factory = Over, Under
