"""
A diagram as an `anywidget <https://anywidget.dev>`_, i.e. a widget of any
frontend that speaks the Jupyter widgets protocol: Jupyter, marimo, VS Code.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    DiagramWidget

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> f = Box('f', Ty('x'), Ty('y'))
>>> widget = f.to_widget()  # doctest: +EXTRA
>>> assert isinstance(widget, DiagramWidget)
>>> assert widget.svg == f.to_svg()
"""

import anywidget
import traitlets


class DiagramWidget(anywidget.AnyWidget):
    """
    A widget showing the SVG of a drawing, synced with its ``svg`` trait.

    Parameters:
        svg : The SVG to show, e.g. from :meth:`RichDisplay.to_svg`.
    """
    _esm = """
    function render({ model, el }) {
      el.style.cssText = `
        display: inline-block;
      `;
      let update = () => { el.innerHTML = model.get("svg") || ""; };
      model.on("change:svg", update);
      update();
    }
    export default { render };
    """
    _css = """
    .jupyter-widgets.anywidget-widget { display: inline-block; }
    """
    svg = traitlets.Unicode("").tag(sync=True)
