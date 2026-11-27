"""
Make inheritance unambiguous with as short a name as possible.

Every base is shown by its short name, unless it shares it with another base
or with the class itself: then it is qualified only as far as it takes to
tell them apart, e.g. the bases of `discopy.ribbon.Diagram` read
`pivotal.Diagram, balanced.Diagram, RibbonCategory` and those of
`discopy.symmetric.Box` read `balanced.Box, Diagram`.
"""


def qualify(names):
    """ The shortest dotted suffix of each name that no other name ends in. """
    def suffix(name):
        parts = name.split(".")
        for k in range(1, len(parts)):
            if not any(other != name and other.split(".")[-k:] == parts[-k:]
                       for other in names):
                return ".".join(parts[-k:])
        return name
    return [suffix(name) for name in names]


def process_bases(app, name, obj, options, bases):
    if isinstance(obj, type):
        classes = [getattr(base, "__origin__", base) for base in bases]
        names = [f"{cls.__module__}.{cls.__qualname__}" for cls in classes
                 if isinstance(cls, type)]
        shown = dict(zip(names, qualify(names + [
            f"{obj.__module__}.{obj.__qualname__}"])))
        for i, cls in enumerate(classes):
            if not isinstance(cls, type):
                continue
            full = f"{cls.__module__}.{cls.__qualname__}"
            if "." in shown[full]:
                bases[i] = f":class:`{shown[full]} <{full}>`"


def setup(app):
    app.connect("autodoc-process-bases", process_bases)

    return {
        'version': '0.2',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
