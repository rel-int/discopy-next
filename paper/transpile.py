"""
Typst to LaTeX, for this paper alone.

The paper is written in the small part of Typst this module reads, and
anything outside it raises :class:`Unsupported` with the offending text,
so that the LaTeX never silently differs from the Typst:

* headings ``=``, ``==``, ``===`` with an optional ``<label>``,
* paragraphs, ``- `` and ``+ `` lists,
* ``*strong*``, ``_emphasis_``, ```code```, ``"quotes"``, ``@key`` for a
  citation or a reference, ``https://`` links, ``#link``, ``#footnote``,
* fenced code, inline ``$math$`` and display ``$ math $``,
* the calls of ``template.typ``: ``#show: paper.with``, ``#drawing``,
  ``#gap``, ``#include`` and ``#bibliography``.

>>> print(inline('A *free* category, see @sec:syntax and @selinger2010.'))
A \\textbf{free} category, see \\cref{sec:syntax} and \\cite{selinger2010}.
>>> print(math('f : x times.o y -> z_(i j)'))
f : x \\otimes y \\to z_{i j}
"""

import re
from pathlib import Path

SYMBOLS = {
    "->": r"\to", "|->": r"\mapsto", "=>": r"\Rightarrow",
    "<=": r"\leq", ">=": r"\geq", "!=": r"\neq", "...": r"\dots",
    "times.o": r"\otimes", "plus.o": r"\oplus",
    "times": r"\times", "compose": r"\circ", "dot": r"\cdot",
    "in": r"\in", "forall": r"\forall", "exists": r"\exists",
    "subset.eq": r"\subseteq", "tilde.eq": r"\simeq", "approx": r"\approx",
    "union": r"\cup", "inter": r"\cap", "emptyset": r"\emptyset",
    "sum": r"\sum", "product": r"\prod", "infinity": r"\infty",
    "id": r"\mathrm{id}", "hom": r"\hom",
    "alpha": r"\alpha", "beta": r"\beta", "lambda": r"\lambda",
    "sigma": r"\sigma", "eta": r"\eta", "epsilon": r"\epsilon",
    "quad": r"\quad", "top": r"\top", "bot": r"\bot",
    "and": r"\wedge", "or": r"\vee", "not": r"\neg"}
FONTS = {
    "bold": r"\mathbf", "cal": r"\mathcal", "bb": r"\mathbb",
    "upright": r"\mathrm", "mono": r"\mathtt", "op": r"\operatorname"}
MATH_TOKEN = re.compile(
    r'"[^"]*"|[A-Za-z]+(?:\.[A-Za-z]+)*|\|->|->|=>|<=|>=|!=|\.\.\.|\s+|.')
TEXT_ESCAPES = {
    "&": r"\&", "%": r"\%", "{": r"\{", "}": r"\}", "^": r"\^{}",
    "_": r"\_"}
CODE_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_",
    "#": r"\#", "%": r"\%", "&": r"\&", "$": r"\$", "^": r"\^{}",
    "~": r"\~{}", "'": r"\textquotesingle{}", "-": r"-{}"}
REFERENCES = ("sec:", "fig:", "lst:", "eq:")
INLINE = re.compile(r"""
    (?P<code>`[^`]*`)
  | (?P<math>\$[^$]*\$)
  | (?P<call>\#(?:link|footnote)\b)
  | (?P<url>https?://[^\s\])]*[^\s\]).,;:])
  | (?P<ref>@[A-Za-z][\w:-]*[\w])
  | (?P<strong>\*[^*\s](?:[^*]*[^*\s])?\*)
  | (?P<emph>(?<![\w\\])_[^_\s](?:[^_]*[^_\s])?_(?!\w))
  | (?P<quote>"[^"]*")
  | (?P<escape>\\[\#*_@$\\<])
  | (?P<hash>\#)
""", re.VERBOSE)
HEADING = re.compile(r"(=+) (.*?)(?: <([\w:-]+)>)?$")
SECTIONS = ("section", "subsection", "subsubsection")
CALL = re.compile(
    r"#(show: paper\.with|drawing|gap|include |bibliography|import )[(\[]?")


class Unsupported(ValueError):
    """ Typst that the paper may not use, since no LaTeX is written for it. """


def balanced(text: str, start: int) -> int:
    """
    The index after the bracket closing the one at ``start``, skipping what
    is inside a string or a code span.

    >>> balanced('(a, "]", [b]) c', 0)
    13
    """
    pairs, stack, i = {"(": ")", "[": "]", "{": "}"}, [], start
    while i < len(text):
        char = text[i]
        if char in '"`' and (not stack or stack[-1] != "[" or char == "`"):
            i = text.index(char, i + 1)
        elif char in pairs:
            stack.append(char)
        elif stack and char == pairs[stack[-1]]:
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    raise Unsupported(f"unbalanced bracket in {text[start:start + 40]!r}")


def code(text: str) -> str:
    """
    A code span in typewriter.

    >>> print(code('f_1 >> g'))
    \\texttt{f\\_1 >{}> g}
    """
    escaped = "".join(CODE_ESCAPES.get(char, char) for char in text)
    return r"\texttt{" + escaped.replace(">>", ">{}>").replace(
        "<<", "<{}<") + "}"


def math(text: str) -> str:
    """
    Typst mathematics as LaTeX's: a name is a symbol of :data:`SYMBOLS` or a
    font of :data:`FONTS`, a string is text, a bracket after ``_`` or ``^``
    is a group.

    >>> print(math('bold(C)(x, y) "if" x <= y'))
    \\mathbf{C}(x, y) \\text{if} x \\leq y
    >>> math('a / b')
    Traceback (most recent call last):
    ...
    paper.transpile.Unsupported: '/' in the mathematics 'a / b'
    """
    tokens, result, i = MATH_TOKEN.findall(text), [], 0
    while i < len(tokens):
        token, following = tokens[i], "".join(tokens[i + 1:])
        if token in FONTS or token in "_^" and following.startswith("("):
            end = balanced(following, 0)
            inside = following[1:end - 1]
            group = inside[1:-1] if token in ("mono", "op") else math(inside)
            result.append(FONTS.get(token, token) + "{" + group + "}")
            i += 1 + len(MATH_TOKEN.findall(following[:end]))
            continue
        if token.startswith('"'):
            result.append(r"\text{" + text_escape(token[1:-1]) + "}")
        elif token in SYMBOLS:
            result.append(SYMBOLS[token])
        elif len(token) > 1 and token[0].isalpha() or token in "/#&%\\{}~$":
            raise Unsupported(f"{token!r} in the mathematics {text!r}")
        else:
            result.append(token)
        i += 1
    return "".join(result)


def text_escape(text: str) -> str:
    """ Plain text with LaTeX's special characters escaped. """
    return "".join(TEXT_ESCAPES.get(char, char) for char in text)


def plain(text: str) -> str:
    """
    What is left of a paragraph between its markup, escaped: a backslash, a
    dollar or a backtick there is markup that was not read.

    >>> print(plain("snake_case, 100% of {it}"))
    snake\\_case, 100\\% of \\{it\\}
    """
    stray = re.search(r"[\\$`]", text)
    if stray:
        raise Unsupported(f"{stray.group()!r} in the text {text!r}")
    return text_escape(text)


def call(text: str, start: int) -> tuple[str, int]:
    """ The LaTeX of the ``#link`` or ``#footnote`` at ``start``, its end. """
    if text.startswith("#footnote[", start):
        end = balanced(text, start + len("#footnote"))
        return r"\footnote{" + inline(
            text[start + len("#footnote["):end - 1]) + "}", end
    if not text.startswith('#link("', start):
        raise Unsupported(text[start:start + 40])
    middle = balanced(text, start + len("#link"))
    url = text[start + len('#link("'):middle - 2]
    if text[middle - 2:middle] != '")' or '"' in url:
        raise Unsupported(text[start:middle])
    if not text.startswith("[", middle):
        return r"\url{" + url + "}", middle
    end = balanced(text, middle)
    return r"\href{" + url.replace("#", r"\#").replace("%", r"\%")\
        + "}{" + inline(text[middle + 1:end - 1]) + "}", end


def inline(text: str) -> str:
    """
    A paragraph's markup as LaTeX.

    >>> print(inline('The `Box` class, _i.e._ "a box", at $x -> y$.'))
    The \\texttt{Box} class, \\emph{i.e.} ``a box'', at $x \\to y$.
    """
    result, i = [], 0
    while i < len(text):
        match = INLINE.search(text, i)
        if match is None:
            break
        result.append(plain(text[i:match.start()]))
        kind, found, i = match.lastgroup, match.group(), match.end()
        if kind == "code":
            result.append(code(found[1:-1]))
        elif kind == "math":
            result.append("$" + math(found[1:-1]) + "$")
        elif kind == "call":
            latex, i = call(text, match.start())
            result.append(latex)
        elif kind == "url":
            result.append(r"\url{" + found + "}")
        elif kind == "ref":
            command = "cref" if found[1:].startswith(REFERENCES) else "cite"
            result.append("\\" + command + "{" + found[1:] + "}")
        elif kind == "strong":
            result.append(r"\textbf{" + inline(found[1:-1]) + "}")
        elif kind == "emph":
            result.append(r"\emph{" + inline(found[1:-1]) + "}")
        elif kind == "quote":
            result.append("``" + inline(found[1:-1]) + "''")
        elif kind == "escape":
            result.append({"\\": r"\textbackslash{}", "<": "<"}.get(
                found[1], "\\" + found[1]))
        else:
            raise Unsupported(f"{text[match.start():match.start() + 40]!r}")
    return "".join(result) + plain(text[i:])


def arguments(text: str) -> dict[str, str]:
    """
    The named arguments of a call, each one the content of its brackets or
    its string.

    >>> arguments('title: [A _title_], authors: ("a", "b"),')
    {'title': 'A _title_', 'authors': '"a", "b"'}
    """
    result, i = {}, 0
    for match in re.finditer(r"([a-z]+): (?=[\[(\"])", text):
        if match.start() < i:
            continue
        i = balanced(text, match.end()) if text[match.end()] != '"'\
            else text.index('"', match.end() + 1) + 1
        result[match.group(1)] = text[match.end() + 1:i - 1]
    return result


def blocks(source: str) -> list[str]:
    """
    The blocks of a document: a heading, a fence, a call with its brackets
    and its label, or the lines up to a blank one.

    >>> blocks('= A <sec:a>\\n\\nSome\\ntext.\\n\\n#gap[\\n  To do.\\n]\\n')
    ['= A <sec:a>', 'Some\\ntext.', '#gap[\\n  To do.\\n]']
    """
    result, i = [], 0
    while i < len(source):
        call_match = CALL.match(source, i)
        if source[i] == "\n":
            i += 1
            continue
        if source.startswith("```", i):
            j = source.index("\n```", i) + len("\n```")
        elif call_match and source[call_match.end() - 1] in "([":
            j = balanced(source, call_match.end() - 1)
            j = source.find("\n", j) % (len(source) + 1)
        elif call_match or source.startswith(("=", "//"), i):
            j = source.find("\n", i) % (len(source) + 1)
        else:
            j = source.find("\n\n", i) % (len(source) + 1)
            if re.search(r"^(```|#(?!link|footnote)|=+ )",
                         source[i + 1:j], re.MULTILINE):
                raise Unsupported(
                    f"no blank line after {source[i:i + 40]!r}")
        if not source.startswith("//", i):
            result.append(source[i:j].rstrip())
        i = j
    return result


def items(block: str, marker: str) -> str:
    """ A list, each item a ``marker`` line and its indented lines. """
    name = {"- ": "itemize", "+ ": "enumerate"}[marker]
    entries = re.split(rf"^{re.escape(marker)}", block, flags=re.MULTILINE)
    body = "\n".join(
        r"\item " + inline(" ".join(entry.split())) for entry in entries[1:])
    return f"\\begin{{{name}}}\n{body}\n\\end{{{name}}}"


def figure(block: str) -> str:
    """ A ``#drawing``: the TikZ picture of that name, with its caption. """
    name = re.match(r'#drawing\("(\w+)"', block).group(1)
    label = re.search(r"<([\w:-]+)>\s*$", block)
    caption = arguments(block).get("caption")
    return "\n".join(filter(None, [
        r"\begin{figure}[ht]", r"\centering",
        rf"\input{{figures/{name}.tikz}}",
        caption and r"\caption{" + inline(" ".join(caption.split())) + "}",
        label and r"\label{" + label.group(1) + "}",
        r"\end{figure}"]))


def front(block: str) -> str:
    """ The title, the authors and the abstract. """
    given = arguments(block)
    authors = re.findall(r'"([^"]*)"', given["authors"])
    return "\n".join([
        r"\title{" + inline(given["title"]) + "}",
        r"\author{" + r" \and ".join(map(inline, authors)) + "}",
        r"\date{}", r"\maketitle", r"\begin{abstract}",
        inline(" ".join(given["abstract"].split())), r"\end{abstract}"])


def block_to_latex(block: str, root: Path) -> str:
    """ One block of :func:`blocks` as LaTeX. """
    if block.startswith("```"):
        language, _, listing = block[3:-4].partition("\n")
        option = "[language=Python]" if language in ("python", "py") else ""
        return "\n".join([
            r"\begin{lstlisting}" + option, listing, r"\end{lstlisting}"])
    if block.startswith("="):
        heading = HEADING.match(block)
        if heading is None or len(heading.group(1)) > len(SECTIONS):
            raise Unsupported(block[:60])
        level, title, label = heading.groups()
        return "\\" + SECTIONS[len(level) - 1] + "{" + inline(title) + "}"\
            + (r"\label{" + label + "}" if label else "")
    if block.startswith(("- ", "+ ")):
        return items(block, block[:2])
    if block.startswith("$ ") and block.endswith(" $"):
        return "\\[\n" + math(block[2:-2].strip()) + "\n\\]"
    if block.startswith("#show: paper.with"):
        return front(block)
    if block.startswith("#drawing("):
        return figure(block)
    if block.startswith("#gap["):
        return r"\begin{gap}" + "\n" + inline(
            " ".join(block[len("#gap["):block.rindex("]")].split()))\
            + "\n" + r"\end{gap}"
    if block.startswith("#include "):
        return body((root / block.split('"')[1]).read_text("utf-8"), root)
    if block.startswith("#bibliography("):
        name = block.split('"')[1].removesuffix(".bib")
        return "\\bibliographystyle{plainurl}\n\\bibliography{" + name + "}"
    if block.startswith("#import "):
        return ""
    if block.startswith("#"):
        raise Unsupported(block[:60])
    return inline(" ".join(block.split()))


def body(source: str, root: Path) -> str:
    """ A Typst source as the body of a LaTeX document. """
    return "\n\n".join(filter(None, (
        block_to_latex(block, root) for block in blocks(source))))


def transpile(path: Path) -> str:
    """
    The LaTeX document of the Typst source at ``path``: the body between
    ``preamble.tex``, which sits beside it, and the end of the document.
    """
    path = Path(path)
    preamble = (path.parent / "preamble.tex").read_text("utf-8")
    return "\n".join([
        preamble, r"\begin{document}", "",
        body(path.read_text("utf-8"), path.parent), "", r"\end{document}", ""])
