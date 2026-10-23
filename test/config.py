from discopy.config import *
from discopy.monoidal import Box, Ty


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
