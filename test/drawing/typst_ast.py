"""Tests for the Typst AST."""

import pytest


def test_typst_ast_nodes():
    """The AST nodes render the Typst syntax they stand for."""
    from discopy.drawing.typst_ast import Import, TypstNode
    assert Import("cetz", members=["canvas"]).render() \
        == '#import "cetz": canvas'
    assert Import("fletcher", alias="f").render() == '#import "fletcher" as f'
    assert Import("cetz").render() == '#import "cetz"'
    with pytest.raises(NotImplementedError):
        TypstNode().render()


def test_typst_import_quotes_its_path():
    from discopy.drawing.typst_ast import Import
    assert Import('a"b').render() == '#import "a' + chr(92) + '"b"'


def test_typst_strings_and_imports():
    from discopy.drawing.typst_ast import Import, Str
    assert Str("a\nb\tc").render() == '"a' + chr(92) + 'nb' + chr(92) + 'tc"'
    with pytest.raises(ValueError):
        Import("x", members=["a"], alias="y").render()
