from discopy.grammar.cfg import Word, Rule, Ty
from discopy.grammar.dependency import doc2tree


class Token:
    """ A spaCy token, whose ``children`` is a fresh generator each time. """
    def __init__(self, text, dep, *tokens):
        self.text, self.dep_, self.tokens = text, dep, tokens

    @property
    def children(self):
        return iter(self.tokens)


def test_doc2tree():
    me, him = Token('I', 'nsubj'), Token('him', 'dobj')
    nsubj, dobj, root = Ty('nsubj'), Ty('dobj'), Ty('ROOT')
    assert isinstance(doc2tree(me), Word)
    assert doc2tree(Token('saw', 'ROOT', me, him)) == Rule(
        nsubj @ dobj, root, name='saw')(Word('I', nsubj), Word('him', dobj))
