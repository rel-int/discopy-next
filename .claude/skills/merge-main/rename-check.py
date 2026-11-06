"""List references to Python names that main removed or renamed.

Usage: ``python3 -I rename-check.py [<base-ref> [<main-ref>]]`` from anywhere
in a clone, after merging main. Main defaults to MERGE_HEAD while a merge is
not committed, to HEAD's second parent when HEAD is a merge, and to
origin/main otherwise; the base to the merge base of main and our side. A
module under ``discopy/`` that main deleted or renamed, and a top-level name
of a module that main removed, are looked for in every tracked file of the
working tree but CHANGELOG.md: imports, ``discopy.module`` paths and
``module.Name`` attributes. Each hit prints ``path:line old → new?``, ``?``
alone when no successor was found, and the exit status is 1 when there is
any.
"""
import ast
import difflib
import os
import re
import subprocess
import sys

PACKAGE = "discopy"
SKIP = {"CHANGELOG.md"}
IMPORT = re.compile(
    r"from\s+(\.*[\w.]*)\s+import\s+(\([^)]*\)|[^\n]*)")


def git(*args, check=True):
    """The standard output of a git command."""
    proc = subprocess.run(["git", *args], capture_output=True, check=False)
    if check and proc.returncode:
        sys.exit(proc.stderr.decode().strip())
    return proc.stdout.decode(errors="replace") if not proc.returncode \
        else None


def refs(argv):
    """The base and main revisions to compare."""
    pending = git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False)
    merge = git("rev-parse", "-q", "--verify", "HEAD^2", check=False)
    ours, main = ("HEAD", "MERGE_HEAD") if pending else (
        "HEAD^1", "HEAD^2") if merge else ("HEAD", "origin/main")
    main = argv[1] if len(argv) > 1 else main
    base = argv[0] if argv else git("merge-base", ours, main).strip()
    return base, main


def module_of(path):
    """The dotted module name of a Python file of the package."""
    parts = path[:-3].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def top_names(source):
    """The names a module defines or imports at its top level."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    names, todo = set(), list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.If, ast.Try)):
            todo.extend(node.body + node.orelse + getattr(
                node, "finalbody", []) + [n for h in getattr(
                    node, "handlers", []) for n in h.body])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0]
                         for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target)
                             if isinstance(n, ast.Name))
    return {name for name in names if not name.startswith("__")}


def show(rev, path):
    """A file at a revision, or None."""
    return git("show", f"{rev}:{path}", check=False)


def removals(base, main):
    """Map each removed module or ``module.Name`` to a guessed successor."""
    gone, changed, deleted, new_files = {}, [], [], []
    diff = git("-c", "core.quotePath=false", "diff", "--find-renames",
               "--name-status", base, main, "--", f"{PACKAGE}/")
    for row in diff.splitlines():
        status, *paths = row.split("\t")
        if not paths[0].endswith(".py"):
            continue
        if status.startswith("R"):
            gone[module_of(paths[0])] = module_of(paths[1])
            changed.append((paths[0], paths[1]))
        elif status == "D":
            gone[module_of(paths[0])] = None
            deleted.append(paths[0])
        elif status == "A":
            new_files.append(paths[0])
        elif status == "M":
            changed.append((paths[0], paths[0]))
    for old in deleted:
        gone[module_of(old)] = heir(base, main, old, new_files)
    added = {}
    for old, new in changed:
        before = top_names(show(base, old) or "")
        after = top_names(show(main, new) or "")
        if before is None or after is None:
            continue
        added[module_of(new)] = after - before
        for name in sorted(before - after):
            gone.setdefault(f"{module_of(old)}.{name}",
                            (module_of(new), name))
    for key, value in list(gone.items()):
        if isinstance(value, tuple):
            gone[key] = successor(value, added)
    return gone


def heir(base, main, old, new_files):
    """The added module sharing most top-level names with a deleted one."""
    before = top_names(show(base, old) or "") or set()
    scores = []
    for new in new_files:
        after = top_names(show(main, new) or "") or set()
        if before and after:
            scores.append((len(before & after) / len(before | after),
                           module_of(new)))
    score, module = max(scores, default=(0, None))
    return module if score >= 0.3 else None


def successor(value, added):
    """The likeliest new name of a removed one, or None."""
    module, name = value
    moved = [f"{other}.{name}" for other, names in added.items()
             if name in names and other != module]
    if moved:
        return moved[0]
    close = difflib.get_close_matches(name, added.get(module, ()), 1, 0.5)
    close = close or [new for new in sorted(added.get(module, ()))
                      if name in new]
    return f"{module}.{close[0]}" if close else None


def still_there(gone):
    """Drop what the merged tree defines again."""
    for key in list(gone):
        path = key.replace(".", "/")
        if os.path.exists(path + ".py") or os.path.exists(path):
            del gone[key]
            continue
        module, _, name = key.rpartition(".")
        file = next((f for f in (module.replace(".", "/") + ".py",
                                 module.replace(".", "/") + "/__init__.py")
                     if os.path.exists(f)), None)
        if file is None:
            continue
        with open(file, encoding="utf-8", errors="replace") as handle:
            names = top_names(handle.read())
        if names and name in names:
            del gone[key]
    return gone


def resolve(path, module):
    """The absolute name of a possibly relative import in a file."""
    dots = len(module) - len(module.lstrip("."))
    if not dots:
        return module
    package = module_of(path).split(".")
    if not path.endswith("__init__.py"):
        package = package[:-1]
    package = package[:len(package) - dots + 1]
    return ".".join(package + [module[dots:]] if module[dots:] else package)


def patterns(gone):
    """One regex per removed thing, over dotted paths and attributes."""
    out = []
    for old in gone:
        text = re.escape(old)
        out.append((old, re.compile(rf"(?<![\w.]){text}(?!\w)")))
        module, _, name = old.rpartition(".")
        if module.startswith(PACKAGE + ".") and name and \
                module not in gone and "." in module:
            last = re.escape(module.rsplit(".", 1)[1])
            out.append((old, re.compile(
                rf"(?<![\w]){last}\.{re.escape(name)}(?!\w)")))
    return out


def scan(path, text, gone, regexes):
    """The (line, old) references of one file."""
    hits = set()
    for match in IMPORT.finditer(text):
        source = resolve(path, match.group(1)) if path.endswith(".py") \
            else match.group(1)
        names = re.finditer(r"[\w]+(?:\s+as\s+\w+)?", match.group(2))
        for item in names:
            name = item.group().split()[0]
            offset = match.start(2) + item.start()
            line = text.count("\n", 0, offset) + 1
            for old in (f"{source}.{name}", source):
                if old in gone:
                    hits.add((line, old))
                    break
    for number, line in enumerate(text.splitlines(), 1):
        for old, regex in regexes:
            if regex.search(line):
                hits.add((number, old))
    return hits


def main(argv):
    """Print one line per reference and exit 1 when there is any."""
    os.chdir(git("rev-parse", "--show-toplevel").strip())
    base, main_ref = refs(argv)
    gone = still_there(removals(base, main_ref))
    if not gone:
        return 0
    regexes, count = patterns(gone), 0
    for path in git("ls-files", "-z").split("\0"):
        if not path or path in SKIP or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            continue
        hits = scan(path, text, gone, regexes)
        for line, old in sorted(hits):
            if any(o != old and o.startswith(old + ".") and l == line
                   for l, o in hits):
                continue
            new = gone[old]
            print(f"{path}:{line} {old} → {new + '?' if new else '?'}")
            count += 1
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
