import pytest

pytest.importorskip("spacy")

from discopy.lambek.tokenise import tokenise


@pytest.mark.parametrize("text, words", [
    ("Alice doesn't love Bob's cat.", "Alice does n't love Bob 's cat ."),
    ("A New York-based, so-called 64-year-old pianist-comedian",
     "A New York-based , so-called 64-year-old pianist-comedian"),
    ("paid--really--a lot: $5.2 million, 3% in 1986-87",
     "paid -- really -- a lot : $ 5.2 million , 3 % in 1986-87"),
    ("summer/winter, 1/2", r"summer\/winter , 1\/2"),
    ('He said "go" ("now").', "He said `` go '' -LRB- `` now '' -RRB- ."),
    ("[a] {b}", "-LSB- a -RSB- -LCB- b -RCB-"),
])
def test_tokenise(text, words):
    assert tokenise(text) == [words.split()]


def test_sentences():
    assert tokenise("Mr. Smith left the U.S. at 3 p.m. today.\n\nDid he?") == [
        ["Mr.", "Smith", "left", "the", "U.S.", "at", "3", "p.m.", "today",
         "."], ["Did", "he", "?"]]
    assert tokenise("") == []
