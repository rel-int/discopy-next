# -*- coding: utf-8 -*-

"""
A dependency grammar is both a pregroup and a context-free grammar.

Summary
-------
.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        from_spacy
"""

from discopy.grammar.cfg import Word, Rule
from discopy.monoidal import Ty


def from_spacy(doc):
    """
    Interface with SpaCy's dependency parser, returns a :class:`cfg.Tree`.

    Parameters:
        doc : Spacy Doc object
    """
    root = find_root(doc)
    return doc2tree(root)


def find_root(doc):
    for token in doc:
        if token.dep_ == 'ROOT':
            return token


def doc2tree(root):
    children = list(root.children)
    if not children:
        return Word(root.text, Ty(root.dep_))
    dom = Ty().tensor(*[Ty(child.dep_) for child in children])
    box = Rule(dom, Ty(root.dep_), name=root.text)
    return box(*map(doc2tree, children))
