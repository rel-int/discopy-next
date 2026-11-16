import pickle
import warnings
from importlib import reload

from discopy import lambek
from discopy.utils import dumps, loads


def test_grammar_is_the_deprecated_name_of_lambek():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from discopy import grammar
        grammar = reload(grammar)
        import discopy.grammar.categorial as categorial
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    for name in ("thue", "cfg", "categorial", "pregroup", "dependency",
                 "abstract", "proofnet"):
        assert getattr(grammar, name) is getattr(lambek, name)
    assert categorial is lambek.categorial


def test_data_written_under_the_old_name_reads_back():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import discopy.grammar  # noqa: F401
    n = lambek.pregroup.Ty('n')
    word = lambek.pregroup.Word('Alice', n)
    old = dumps(word).replace("lambek.pregroup", "grammar.pregroup")
    assert old != dumps(word) and loads(old) == word
    written = pickle.dumps(word, protocol=0)
    renamed = written.replace(
        b"discopy.lambek.pregroup", b"discopy.grammar.pregroup")
    assert renamed != written and pickle.loads(renamed) == word
