import re

from discopy.config import *
from discopy.monoidal import Box, Equation, Ty


def test_darken():
    assert darken("#ffffff") == "#999999"
    assert darken("#000000") == "#000000"
    assert darken("#808080", factor=.5) == "#404040"


def test_COLORS_have_a_dark_shade():
    for name, hexcode in COLORS.items():
        if not name.startswith("dark_"):
            assert COLORS[f"dark_{name}"] == darken(hexcode)


def test_BOX_DRAWING_ATTRIBUTES_are_the_defaults_of_a_drawn_box():
    box = Box('f', Ty('x'), Ty('y'), color="red")
    drawn, = box.to_drawing().boxes
    for attribute, default in BOX_DRAWING_ATTRIBUTES.items():
        expected = "red" if attribute == "color" else default(drawn)
        assert getattr(drawn, attribute) == expected, attribute


def label_sizes(diagram, **params):
    """ The pairs of a label and its size in points in a diagram's SVG. """
    labels = re.findall(
        r'<!-- (.*?) -->\s*<g transform="[^"]*scale\(([\d.]+) ',
        diagram.to_svg(**params))
    return {(text, 100 * float(scale)) for text, scale in labels}


def test_DRAWING_DEFAULT_fontsize_is_the_size_of_box_and_wire_labels():
    f = Box('f', Ty('x'), Ty('y'))
    equation = Equation(f, f)
    default = DRAWING_DEFAULT["fontsize"]
    spiders = DRAWING_DEFAULT["fontsize_spiders"]
    assert (default, spiders) == (10, 12)
    assert label_sizes(equation) == {
        ('x', default), ('y', default), ('f', default), ('=', spiders)}
    assert label_sizes(equation, fontsize=20, fontsize_types=8) == {
        ('x', 8), ('y', 8), ('f', 20), ('=', spiders)}
    assert equation.to_svg() == equation.to_svg(
        fontsize=default, fontsize_types=default)
