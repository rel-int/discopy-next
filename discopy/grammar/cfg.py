# -*- coding: utf-8 -*-

"""
A context free grammar is a formal grammar where the rules all have a codomain
of length 1.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Tree
    Rule
    Word
    Id
    Operad
    Algebra

Axioms
------

The axioms of multicategories (aka operads) hold on the nose.

>>> x, y = Ty('x'), Ty('y')
>>> f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
>>> h = Rule(y @ x, x, name='h')
>>> assert f(g, h) == Tree(f, *[g, h])

>>> assert Id(x)(f) == f == f(Id(x), Id(x))
>>> left = f(Id(x), h)(g, Id(x), Id(x))
>>> right = f(g, Id(x))(Id(x), Id(x), h)
>>> assert f(g, h) == left == right
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from discopy import monoidal
from discopy.cat import factory, Functor
from discopy.config import BOX_DRAWING_ATTRIBUTES
from discopy.grammar import thue
from discopy.monoidal import Ty
from discopy.utils import (
    assert_isinstance, factory_name, assert_isatomic, AxiomError)

if TYPE_CHECKING:
    import nltk


@factory
class Tree:
    """
    A tree is a rule for the ``root`` and a list of trees called ``branches``,
    its ``dom`` is the tensor of theirs, as composition in the free operad.

    Example
    -------

    We build a syntax tree from a context-free grammar.

    >>> n, d, v = Ty('N'), Ty('D'), Ty('V')
    >>> vp, np, s = Ty('VP'), Ty('NP'), Ty('S')
    >>> Caesar, crossed = Word('Caesar', n), Word('crossed', v)
    >>> the, Rubicon = Word('the', d), Word('Rubicon', n)
    >>> VP, NP = Rule(n @ v, vp), Rule(d @ n, np)
    >>> S = Rule(vp @ np, s)
    >>> sentence = S(VP(Caesar, crossed), NP(the, Rubicon))
    """
    ob = Ty

    def __init__(self, root: Rule, *branches: Tree):
        assert_isinstance(root, Rule)
        for branch in branches:
            assert_isinstance(branch, Tree)
        if not isinstance(self, Rule) and not root.dom == Ty().tensor(
                *[branch.cod for branch in branches]):
            raise AxiomError
        self.cod, self.root, self.branches = root.cod, root, branches
        if not isinstance(self, Rule):
            self.dom = Ty().tensor(*[branch.dom for branch in branches])

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, (self.root, ) + self.branches))})"

    def __str__(self):
        return f"{self.root}({', '.join(map(str, self.branches))})"

    def __call__(self, *others):
        if not others or all([isinstance(other, Id) for other in others]):
            return self
        if isinstance(self, Id):
            return others[0]
        if isinstance(self, Rule):
            return Tree(self, *others)
        if isinstance(self, Tree):
            lengths = [len(branch.dom) for branch in self.branches]
            ranges = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]
            branches = [self.branches[i](*others[ranges[i]:ranges[i + 1]])
                        for i in range(len(self.branches))]
            return Tree(self.root, *branches)
        raise NotImplementedError()

    @staticmethod
    def id(dom):
        return Id(dom)

    def __eq__(self, other):
        return isinstance(other, Tree)\
            and (self.root, self.branches) == (other.root, other.branches)

    def __hash__(self):
        return hash((self.root, self.branches)) if self.branches\
            else hash(self.root)

    def to_diagram(self) -> monoidal.Diagram:
        """
        Interface between Tree and monoidal.Diagram.

        >>> x = Ty('x')
        >>> f = Rule(x @ x, x, name='f')
        >>> tree = f(f(f, f), f)
        >>> tree.to_diagram().foliation().draw(
        ...     doctest='docs/_static/grammar/tree-to-diagram.svg')

        .. image:: /_static/grammar/tree-to-diagram.svg
            :align: center
        """
        return self.root.to_diagram()\
            << monoidal.Id().tensor(*[t.to_diagram() for t in self.branches])

    @staticmethod
    def from_nltk(tree: nltk.Tree) -> Tree:
        """
        Interface with NLTK

        >>> import nltk  # doctest: +EXTRA
        >>> t = nltk.Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
        >>> print(Tree.from_nltk(t))
        S(I, VP(saw, him))
        >>> Tree.from_nltk(t).branches[0]
        grammar.cfg.Word('I', monoidal.Ty(monoidal.Wire('NP')))

        A word is typed by the label of the tree it sits in.

        >>> print(Tree.from_nltk(nltk.Tree.fromstring("(NP the big dog)")))
        NP(the, big, dog)
        """
        label = tree.label()
        if len(tree) == 1 and isinstance(tree[0], str):
            return Word(tree[0], Ty(label))
        branches = [
            Word(branch, Ty(label)) if isinstance(branch, str)
            else Tree.from_nltk(branch) for branch in tree]
        dom = Ty().tensor(*[branch.cod for branch in branches])
        return Rule(dom, Ty(label), name=label)(*branches)


class Rule(Tree, thue.Rule):
    """
    A rule is a generator of free operads, given by a type ``dom`` of
    arbitrary length, an atomic type ``cod`` and an optional ``name``.

    Its dagger is a rule with an atomic ``dom`` instead.

    Parameters:
        dom : The domain of the rule, i.e. its input.
        cod : The codomain of the rule, i.e. its output.
        name : The name of the rule.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Rule(x @ y, x, name='f')
    >>> assert (f[::-1].dom, f[::-1].cod) == (x, x @ y)
    >>> assert f[::-1][::-1] == f != f[::-1]
    """
    def __init__(self, dom: monoidal.Ty, cod: monoidal.Ty, name: str = None,
                 **params):
        assert_isinstance(dom, Ty)
        assert_isatomic(dom if params.get('is_dagger', False) else cod, Ty)
        thue.Rule.__init__(self, dom=dom, cod=cod, name=name, **params)
        Tree.__init__(self, root=self)

    def __eq__(self, other):
        if isinstance(other, Rule):
            return thue.Rule.__eq__(self, other)
        return isinstance(other, Tree)\
            and other.root == self and not other.branches

    __hash__, __str__ = thue.Rule.__hash__, thue.Rule.__str__
    __repr__ = thue.Rule.__repr__

    def to_diagram(self) -> monoidal.Box:
        drawing = {attr: getattr(self, attr)
                   for attr in BOX_DRAWING_ATTRIBUTES if hasattr(self, attr)}
        return monoidal.Box(self.name, self.dom, self.cod, data=self.data,
                            is_dagger=self.is_dagger, **drawing)


class Word(thue.Word, Rule):
    """
    A word is a leaf in a context-free tree.

    Parameters:
        name : The name of the word.
        cod : The grammatical type of the word.
        dom : An optional domain for the word, empty by default.
    """
    def __init__(self, name: str, cod: monoidal.Ty, dom: monoidal.Ty = Ty(),
                 **params):
        thue.Word.__init__(self, name=name, dom=dom, cod=cod, **params)
        Rule.__init__(self, dom=dom, cod=cod, name=name, **params)


class Id(Rule):
    """ The identity is a rule that does nothing. """
    serialised_attrs = ("dom", )

    def __init__(self, dom):
        Rule.__init__(self, dom, dom, name=f"Id({dom})")

    def __repr__(self):
        return factory_name(type(self)) + f"({self.dom!r})"

    def dagger(self) -> Id:
        return self


class Operad:
    """
    An operad is a category with a method ``__call__`` which constructs a tree
    from a root and a list of branches.

    Parameters:
        ob : The colours of the operad.
        ar : The operations of the operad.
    """
    ob = Ty
    ar = Tree


class Algebra(Functor):
    """
    An algebra is a functor with the free operad as domain and a given operad
    as codomain.

    Parameters:
        ob_map (dict[monoidal.Ty, cod.ob]) :
            The mapping from domain to codomain colours.
        ar_map (dict[Rule, cod]):
            The mapping from domain to codomain operations.
        cod (Operad) : The codomain of the algebra.
    """
    dom = cod = Operad()

    def __call__(self, other):
        if isinstance(other, Id):
            return self.cod.id(self.ob_map[other.dom])
        if isinstance(other, Rule):
            return self.ar_map[other]
        if isinstance(other, Tree):
            return self(other.root)(*map(self, other.branches))
        raise TypeError
