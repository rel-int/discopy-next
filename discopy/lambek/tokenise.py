# -*- coding: utf-8 -*-

r"""
The words of English text as the Penn Treebank writes them, which is how
CCGbank and CCGrebank write them, so that a parser trained on either, e.g.
:class:`discopy.lambek.categorial.bobcat.Parser`, reads raw text. It is the
rule-based tokenizer and sentencizer of `spaCy <https://spacy.io>`_, with no
statistical model, as lambeq's ``SpacyTokeniser`` is, and it needs spaCy,
which the ``lambek`` extra installs.

spaCy already splits ``n't``, ``'s``, ``'re``, ``$``, ``%`` and ``--`` as
the treebank does and keeps abbreviations such as ``U.S.`` whole. Where it
differs, the treebank wins:

* a hyphen or a slash inside a word does not split it, e.g. ``York-based``,
  ``1986-87`` and ``summer\/winter``, and a slash is escaped as ``\/``,
* a double quote is two backquotes when it opens and two single quotes when
  it closes,
* brackets are ``-LRB-``, ``-RRB-``, ``-LSB-``, ``-RSB-``, ``-LCB-`` and
  ``-RCB-``.

Two differences stay: a single quote is ``'`` and splits from a decade,
e.g. ``' 90s`` where the treebank has ``'90s``, and a sentence ends at a
``.``, ``!`` or ``?`` of its own, so that ``at 3 p.m. Then`` does not end
one. ``can't`` is ``ca n't`` in both.

Summary
-------
.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        pipeline
        word
        tokenise
"""

from functools import cache
from typing import Any

BRACKETS = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", "]": "-RSB-",
            "{": "-LCB-", "}": "-RCB-"}


@cache
def pipeline():
    """
    spaCy's blank English pipeline with a sentencizer, whose infixes split a
    word on a dash but not on a hyphen or a slash.
    """
    # pylint: disable=import-outside-toplevel  # spaCy is an extra
    from spacy.lang.char_classes import HYPHENS  # pylint: disable=import-error
    from spacy.lang.en import English  # pylint: disable=import-error
    from spacy.util import compile_infix_regex  # pylint: disable=import-error

    nlp = English()
    nlp.add_pipe("sentencizer")
    infixes = [rule.replace(f"(?:{HYPHENS})", "(?:--|---|–|—|——)")
               .replace(r"[+\-\*^]", r"[+\*^]").replace("[:<>=/]", "[:<>=]")
               for rule in nlp.Defaults.infixes]
    tokenizer: Any = nlp.tokenizer  # spaCy types it as a callable
    tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    return nlp


def word(token) -> str:
    """
    The treebank's word for a spaCy token: a double quote opens at the start,
    after a space or after an opening bracket, and closes otherwise.

    Parameters:
        token : The token.
    """
    if token.text == '"':
        opens = token.i == 0 or token.nbor(-1).whitespace_\
            or token.nbor(-1).text in "([{"
        return "``" if opens else "''"
    return BRACKETS.get(token.text, token.text.replace("/", "\\/"))


def tokenise(text: str) -> list[list[str]]:
    r"""
    The words of each sentence of a text.

    Parameters:
        text : The text.

    Example
    -------
    >>> text = 'Alice isn\'t "New York-based" (yet). Is Bob?'
    >>> for words in tokenise(text):  # doctest: +EXTRA
    ...     print(*words)
    Alice is n't `` New York-based '' -LRB- yet -RRB- .
    Is Bob ?
    """
    return [[word(token) for token in sentence if not token.is_space]
            for sentence in pipeline()(text).sents]
