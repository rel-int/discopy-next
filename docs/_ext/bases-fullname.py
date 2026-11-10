"""
Make inheritance unambiguous by showing full class name.

Every base from another module is shown by its full name, e.g. the bases of
`discopy.symmetric.Box` read `discopy.balanced.Box, Diagram`.
"""

def process_bases(app, name, obj, options, bases):
    if isinstance(obj, type):
        ambiguity = set(base for base in obj.__bases__ if base.__module__ is not obj.__module__)
        for i, base in enumerate(bases):
            if base in ambiguity:
                bases[i] = ":class:`{}.{}`".format(base.__module__, base.__name__)


def setup(app):
    app.connect("autodoc-process-bases", process_bases)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
