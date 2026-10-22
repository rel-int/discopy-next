from discopy import cat, grammar, monoidal
from discopy.grammar.thue import *


def test_Rule_and_Word():
    s, n, v, p = map(Ty, "SNVP")
    rule, word = Rule(n @ p, s), Word('Jane', n, dom=v)
    assert (rule.dom, rule.cod, rule.name) == (n @ p, s, "Rule(N @ P, S)")
    assert (word.dom, word.cod, word.name) == (v, n, 'Jane')
    assert Word('John', n).dom == Ty()
    assert isinstance(word, Rule) and isinstance(rule, monoidal.Box)


def test_sentence():
    s, n, v, p = map(Ty, "SNVP")
    r0, r1 = Rule(n @ p, s), Rule(v @ n, p)
    Jane, loves, John = Word('Jane', n), Word('loves', v), Word('John', n)
    sentence = Jane @ loves @ John >> n @ r1 >> r0
    assert (sentence.dom, sentence.cod) == (Ty(), s)
    assert sentence.boxes == [Jane, loves, John, r1, r0]


def test_repr():
    s, n = Ty('S'), Ty('N')
    scope = {"grammar": grammar, "monoidal": monoidal, "cat": cat}
    for box in [Rule(n, s), Rule(n, s, name='r'), Word('Jane', n),
                Word('Jane', n, dom=s)]:
        assert eval(repr(box), scope) == box
