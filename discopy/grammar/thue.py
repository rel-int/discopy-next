"""
A formal grammar is a free monoidal category with words and rules as boxes.

Formal grammars are also known as string rewriting or semi-Thue systems, they
were introduced by Thue :cite:`Thue14`.

The parsing problem is to decide, given two strings, whether there exists a
diagram from one to the other. It has been shown to be undecidable by Post
:cite:`Post47` and Markov :cite:`Markov47`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Word
    Rule

Example
-------
>>> s, n, v, p = map(Ty, "SNVP")
>>> r0, r1 = Rule(n @ p, s), Rule(v @ n, p)
>>> Jane, loves, John = Word('Jane', n), Word('loves', v), Word('John', n)
>>> sentence = Jane @ loves @ John >> n @ r1 >> r0
>>> sentence.draw(
...     figsize=(4, 3), doctest='docs/_static/grammar/cfg-example.svg')

.. image:: /_static/grammar/cfg-example.svg
    :align: center
"""
from discopy import monoidal
from discopy.monoidal import Ty  # noqa: F401  pylint: disable=unused-import
from discopy.axioms import Serialisable


class Rule(monoidal.Box):
    """
    A rule is a box with monoidal types as ``dom`` and ``cod`` and an
    optional ``name``.

    Parameters:
        dom : The domain of the rule, i.e. its input.
        cod : The codomain of the rule, i.e. its output.
        name : The name of the rule, empty by default.
    """
    serialised_attrs = ('dom', 'cod', 'name', 'is_dagger', 'data')

    def __init__(self, dom: monoidal.Ty, cod: monoidal.Ty, name: str = None,
                 **params):
        name = f"Rule({dom}, {cod})" if name is None else name
        monoidal.Box.__init__(self, name, dom, cod, **params)


class Word(Rule):
    """
    A word is a rule with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.

    Parameters:
        name : The name of the word.
        cod : The grammatical type of the word.
        dom : An optional domain for the word, empty by default.
    """
    serialised_attrs = ('name', 'cod', 'dom', 'is_dagger', 'data')
    __repr__ = Serialisable.__repr__  # Over cfg.Tree's, earlier in the MRO
    # of grammar.cfg.Word.

    def __init__(self, name: str, cod: monoidal.Ty, dom: monoidal.Ty = None,
                 **params):
        dom = self.ob() if dom is None else dom
        Rule.__init__(self, dom=dom, cod=cod, name=name, **params)
