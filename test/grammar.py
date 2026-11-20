import pickle
import subprocess
import sys
from importlib import import_module, reload

from pytest import warns

from discopy import lambek
from discopy.utils import dumps, loads


def test_grammar_is_the_deprecated_name_of_lambek():
    with warns(DeprecationWarning, match="use discopy.lambek instead"):
        grammar = reload(import_module("discopy.grammar"))
    for name in ("thue", "cfg", "categorial", "pregroup", "dependency",
                 "abstract"):
        assert getattr(grammar, name) is getattr(lambek, name)
        assert import_module(f"discopy.grammar.{name}")\
            is getattr(lambek, name)
        assert getattr(lambek, name).__spec__.name == f"discopy.lambek.{name}"
    from discopy.grammar import Word, Rule
    from discopy.grammar.cfg import Tree
    assert (Word, Rule, Tree) == (lambek.Word, lambek.Rule, lambek.cfg.Tree)


def test_the_package_still_names_grammar():
    import discopy
    vars(discopy).pop("grammar", None)
    with warns(DeprecationWarning, match="use discopy.lambek instead"):
        assert discopy.grammar.pregroup is lambek.pregroup


def test_a_tree_of_the_old_name_loads_in_a_fresh_process():
    word = lambek.pregroup.Word('Alice', lambek.pregroup.Ty('n'))
    old = dumps(word).replace("lambek.pregroup", "grammar.pregroup")
    assert old != dumps(word) and loads(old) == word
    script = (
        "import sys; from discopy.utils import dumps, loads; "
        "assert 'discopy.grammar' not in sys.modules; "
        "print(dumps(loads(sys.argv[1])))")
    fresh = subprocess.run(
        [sys.executable, "-c", script, old],
        capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == dumps(word)


def test_a_pickle_of_the_old_name_loads_in_a_fresh_process():
    word = lambek.pregroup.Word('Alice', lambek.pregroup.Ty('n'))
    written = pickle.dumps(word, protocol=0)
    renamed = written.replace(
        b"discopy.lambek.pregroup", b"discopy.grammar.pregroup")
    assert renamed != written
    script = (
        "import pickle, sys, warnings; warnings.simplefilter('ignore'); "
        "from discopy.utils import dumps; "
        "print(dumps(pickle.loads(sys.stdin.buffer.read())))")
    fresh = subprocess.run(
        [sys.executable, "-c", script], input=renamed,
        capture_output=True, check=True)
    assert fresh.stdout.decode().strip() == dumps(word)
