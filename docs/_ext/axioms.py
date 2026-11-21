"""
Draw the axioms of each level of the syntax on its page, so that a law stated
with :func:`discopy.testable.axiom` in :mod:`discopy.abc` draws itself in the
documentation with no extra step.

A class of :mod:`discopy.abc` is abstract: its axioms are drawn in their
:func:`discopy.drawing.bracketing.free_category`, the most general free
category of diagrams that states them, from their generic instance, with the
brackets that tell the sides apart, see
:func:`discopy.drawing.bracketing.equation`. Each axiom is shown twice: on
the page of the class that states it and on the page of the ``Diagram`` it
is drawn in, e.g. :meth:`discopy.abc.MarkovCategory.copy_coassociativity` on
those of :class:`discopy.abc.MarkovCategory` and
:class:`discopy.markov.Diagram`. An axiom only inherited is not repeated
below, and one that no free category of diagrams states, e.g.
:meth:`discopy.abc.Groupoid.invertibility`, is not drawn.

The drawings are generated when the builder starts, as SVG files in
``_static/axioms/`` of the build output rather than committed, and appended
to the docstring of each class under an "Axioms" rubric, with the docstring
of each axiom, for HTML builders only.
"""

import inspect
import os

from sphinx.util import logging
from sphinx.util.osutil import relative_uri

logger = logging.getLogger(__name__)

STATIC = "_static/axioms"


PAGES = {}
""" The axioms each class shows, by the full name of the class. """


def generate(app):
    """ Draw every stated axiom and record which pages show it. """
    from discopy import abc
    from discopy.drawing.bracketing import (
        stated_axioms, free_category, equation)

    PAGES.clear()
    if app.builder.format != "html":
        return
    os.makedirs(os.path.join(app.outdir, STATIC), exist_ok=True)
    for name, cls in vars(abc).items():
        if not isinstance(cls, type) or cls.__module__ != abc.__name__:
            continue
        for law, axiom in stated_axioms(cls).items():
            free = free_category(axiom)
            if free is None:
                continue
            path = f"{STATIC}/{name}.{law}.svg"
            equation(axiom).draw(path=os.path.join(app.outdir, path))
            entry = (f"{name}.{law}", inspect.getdoc(axiom.function), path)
            for page in (cls, free):
                PAGES.setdefault(
                    f"{page.__module__}.{page.__qualname__}", []).append(entry)
    logger.info("[axioms] drew the axioms of %s", ", ".join(PAGES))


def append(app, what, name, obj, options, lines):
    """ Append the drawings of the axioms of a class to its docstring. """
    del obj, options  # the class is known by name
    if what != "class" or name not in PAGES:
        return
    page = app.builder.get_target_uri(app.env.docname)
    lines.extend(["", ".. rubric:: Axioms", ""])
    for law, doc, path in PAGES[name]:
        lines.extend([
            f":meth:`~discopy.abc.{law}`",
            *(f"    {line}" for line in (doc or "").splitlines()), "",
            "    .. raw:: html", "",
            f'        <img src="{relative_uri(page, path)}" alt="{law}"'
            ' style="max-width: 100%">', ""])


def setup(app):
    app.connect("builder-inited", generate)
    app.connect("autodoc-process-docstring", append)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
