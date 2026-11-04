"""Run the Python code blocks of a README, for every suite that checks one."""

import re

CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def run_code_blocks(readme):
    """
    Execute the Python code blocks of ``readme`` in order, in one namespace:
    they share state, e.g. a diagram built in one block is evaluated in the
    next. matplotlib is imported here, not at the top: pytest's ``testpaths``
    glob reaches this file, see :mod:`fixtures.zx_graphs`.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    text = readme.read_text(encoding="utf-8")
    blocks = list(CODE_BLOCK.finditer(text))
    assert blocks, f"no Python code blocks found in {readme}"
    namespace = {}
    for match in blocks:
        # Pad with newlines so tracebacks point at the actual README lines.
        source = "\n" * text.count("\n", 0, match.start(1)) + match.group(1)
        exec(compile(source, str(readme), "exec"), namespace)  # noqa: S102
        plt.close("all")
    return namespace
