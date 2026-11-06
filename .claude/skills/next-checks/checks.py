"""
Run the gates of CONTRIBUTING.md's "Run the unit tests" on the current tree
and print one line per gate, ``PASS|FAIL|SKIP <gate>: <fact>``.

ruff, ty, pylint and a full suite are compared with the merge base of HEAD
and ``origin/main``, measured once in a temporary worktree and cached under
``~/.cache/next-checks/<kind>-<key>/``, the key hashing what the measurement
reads; the tree under check is the working directory, untracked files
included. Findings are matched by (rule, file, message) with the numbers in
the message ignored, so code moving down a few lines is not new. A tool
that fails to run is a FAIL, never cached. Full logs go to a directory
named on the last line. Exit 1 on any FAIL.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

GATES = ("flake8", "github", "pylint", "ruff", "ty", "tests", "notebooks",
         "changelog")
CACHE = pathlib.Path(os.environ.get(
    "NEXT_CHECKS_CACHE", pathlib.Path.home() / ".cache" / "next-checks"))
VENV314 = CACHE / "venv-3.14"
SPECIAL_TESTS = {"discopy/neural/torch.py": "test/neural/torch_module.py"}
NOT_TREE = {"TODO.md", "CHANGELOG.md", ".claude"}
KEEP = 20


class ToolError(Exception):
    pass


def run(cmd, cwd, log, timeout=None):
    env = dict(os.environ, PYTHONPATH=str(cwd), MPLBACKEND="Agg")
    env.pop("VIRTUAL_ENV", None)
    with open(log, "a", encoding="utf-8") as file:
        file.write(f"$ (cd {cwd} && {' '.join(map(str, cmd))})\n")
        file.flush()
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=cwd, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False)
        file.write(proc.stdout + f"\n[exit {proc.returncode}]\n")
    return proc.returncode, proc.stdout


def git(*args, cwd, env=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
        env=None if env is None else dict(os.environ, **env)).stdout.strip()


def first_line(text, pattern=r"\S"):
    return next((line.strip() for line in text.splitlines()
                 if re.search(pattern, line)), "no output")[:160]


def last_error(out):
    errors = re.findall(r"^(?:E\s+)?(\w+(?:Error|Exception): .*)$", out, re.M)
    return errors[-1][:160] if errors else first_line(out[-2000:])


def capped(items, n=5):
    items = list(items)
    more = f", +{len(items) - n}" if len(items) > n else ""
    return ", ".join(items[:n]) + more


class Context:
    def __init__(self, args):
        self.args = args
        self.root = pathlib.Path(git("rev-parse", "--show-toplevel",
                                     cwd=os.getcwd()))
        self.base = git("merge-base", "HEAD", args.base, cwd=self.root)
        git("worktree", "prune", cwd=self.root)
        self.prune()
        self.logs = pathlib.Path(tempfile.mkdtemp(
            prefix=time.strftime("%m%d-%H%M%S-"),
            dir=self.mkdir(CACHE / "logs")))
        self.lock_error = None
        self.bin, self.python, self.fallback = self.venv()
        self.tree = self.working_tree()
        self.changed = sorted(set(filter(None, git(
            "diff", "--name-only", "--no-renames", self.base,
            cwd=self.root).splitlines()
            + git("ls-files", "--others", "--exclude-standard",
                  cwd=self.root).splitlines())))
        self.lock, self.worktree = threading.Lock(), None

    @staticmethod
    def mkdir(path):
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def prune():
        for pattern in ("logs/*", "lint-*", "full-*"):
            old = sorted(CACHE.glob(pattern), key=lambda p: p.stat().st_mtime)
            for path in old[:-KEEP]:
                shutil.rmtree(path, ignore_errors=True)

    def venv(self):
        log = self.logs / "venv.log"
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(VENV314))
        with open(log, "w", encoding="utf-8") as file:
            proc = subprocess.run(
                ["uv", "sync", "--locked", "--group", "dev", "--group", "all",
                 "--python", "3.14"], cwd=self.root, env=env, stdout=file,
                stderr=subprocess.STDOUT, check=False)
        if proc.returncode == 0:
            return VENV314 / "bin", "3.14", None
        text = log.read_text(encoding="utf-8")
        if "needs to be updated" in text:
            self.lock_error = first_line(text, "needs to be updated")
        common = pathlib.Path(git("rev-parse", "--path-format=absolute",
                                  "--git-common-dir", cwd=self.root))
        venv = common.parent / ".venv" / "bin"
        version = subprocess.run(
            [str(venv / "python"), "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, check=False).stdout.strip()
        return venv, version or "unknown", first_line(text, "error")[:80]

    def working_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = pathlib.Path(tmp) / "index"
            real = self.root / git("rev-parse", "--git-path", "index",
                                   cwd=self.root)
            if real.exists():
                shutil.copy(real, index)
            env = {"GIT_INDEX_FILE": str(index)}
            git("add", "-A", "--", ".", cwd=self.root, env=env)
            return git("write-tree", cwd=self.root, env=env)

    def key(self, tree, kind):
        if kind == "lint":
            parts = [git("rev-parse", f"{tree}:{path}", cwd=self.root)
                     for path in ("discopy", ".pylintrc", "pyproject.toml",
                                  "uv.lock")]
        else:
            parts = [line for line in git(
                "ls-tree", tree, cwd=self.root).splitlines()
                if line.split("\t")[-1] not in NOT_TREE
                and not line.split("\t")[-1].startswith(".")]
        parts += [self.python, "v3"]
        digest = hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]
        path = self.mkdir(CACHE / f"{kind}-{digest}")
        os.utime(path)
        return path

    def base_tree(self):
        with self.lock:
            if self.worktree is None:
                path = pathlib.Path(
                    tempfile.mkdtemp(prefix="next-checks-base-"))
                git("worktree", "add", "--detach", "-f", str(path), self.base,
                    cwd=self.root)
                version = self.root / "discopy" / "version.py"
                if version.exists():
                    shutil.copy(version, path / "discopy" / "version.py")
                self.worktree = path
            return self.worktree

    def close(self):
        if self.worktree is not None:
            git("worktree", "remove", "--force", str(self.worktree),
                cwd=self.root)

    def under(self, prefix):
        return [path for path in self.changed if path.startswith(prefix)]


def cached(directory, name, measure):
    path = directory / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = measure()
    if not (isinstance(value, dict) and value.get("interrupted")):
        path.write_text(json.dumps(value), encoding="utf-8")
    return value


def measured(ctx, kind, name, measure):
    head_key = ctx.key(ctx.tree, kind)
    base_key = ctx.key(ctx.base, kind)
    log = ctx.logs / f"{name}.log"
    head = cached(head_key, name, lambda: measure(ctx.root, log))
    if base_key == head_key:
        return head, head
    try:
        base = cached(base_key, name, lambda: measure(ctx.base_tree(), log))
    except ToolError as error:
        raise ToolError(f"on base: {error}") from error
    return head, base


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def gate_flake8(ctx):
    log = ctx.logs / "flake8.log"
    code, out = run([ctx.bin / "pflake8", "discopy"], ctx.root, log)
    findings = [line for line in out.splitlines()
                if re.match(r"\S+:\d+:", line)]
    if code == 0:
        return "PASS", "0 findings"
    return "FAIL", f"{plural(len(findings), 'finding')}, first: " + (
        findings[0] if findings else first_line(out))


def gate_github(ctx):
    if not ctx.under(".github/"):
        return "SKIP", "no change under .github/"
    log, failed = ctx.logs / "github.log", []
    code, out = run([ctx.bin / "pflake8", ".github"], ctx.root, log)
    if code:
        failed.append("pflake8 .github: " + first_line(out, r"^\S+:\d+:"))
    if shutil.which("actionlint"):
        code, out = run(["actionlint"], ctx.root, log)
        if code:
            failed.append("actionlint: " + first_line(out, r"^\S+:\d+:"))
    tests = sorted(map(str, (ctx.root / ".github" / "tests").glob("*.py")))
    code, out = run([ctx.bin / "python", "-m", "pytest", "-q", "-p",
                     "no:cacheprovider", "-rfE", "--import-mode=importlib",
                     *tests], ctx.root, log)
    if code:
        failed.append(".github/tests: " + (failures(out) or [summary(out)])[0])
    note = "" if shutil.which("actionlint") else (
        " (actionlint absent, CI runs it)")
    return ("FAIL", "; ".join(failed) + note) if failed else (
        "PASS", "pflake8 .github, .github/tests" + note)


def pylint_measure(ctx):
    def measure(root, log):
        side = "base" if root != ctx.root else "head"
        out = log.with_name(f"pylint-{side}.json")
        out.unlink(missing_ok=True)
        code, text = run([ctx.bin / "pylint", "discopy", "--output-format",
                          f"json2:{out}"], root, log)
        if code & 32 or not out.exists():
            raise ToolError(f"pylint exited {code}: {first_line(text)}")
        data = json.loads(out.read_text(encoding="utf-8"))
        return {"code": code, "score": data["statistics"]["score"],
                "messages": [[m["symbol"], m["path"], m["line"], m["message"]]
                             for m in data["messages"]]}
    return measure


def gate_pylint(ctx):
    head, base = measured(ctx, "lint", "pylint", pylint_measure(ctx))
    rc = (ctx.root / ".pylintrc").read_text(encoding="utf-8")
    fail_under = float(re.search(r"^fail-under=([\d.]+)", rc, re.M).group(1))
    fail_on = re.search(r"^fail-on=([\w,\-\s]+?)\n\S", rc, re.M)
    fail_on = set(re.findall(r"[\w-]+", fail_on.group(1) if fail_on else ""))
    new = new_items(*([[(s, p, m.split("\n")[0]), line]
                       for s, p, line, m in side["messages"]]
                      for side in (head, base)))
    where = "on 3.14" if ctx.fallback is None else (
        f"on {ctx.python}, 3.14 sync failed, CI lints on 3.14, ±0.02")
    fact = (f"{head['score']:.4f} (base {base['score']:.4f}, fail-under "
            f"{fail_under}, {where}), {plural(len(new), 'new message')}")
    log = ctx.logs / "pylint-head.json"
    if not log.exists():
        log.write_text("".join(json.dumps(m) + "\n"
                               for m in head["messages"]), encoding="utf-8")
    blocking = [m for m in head["messages"] if m[0] in fail_on]
    if head["score"] < fail_under:
        return "FAIL", fact + "; score below fail-under"
    if blocking:
        s, p, line, m = blocking[0]
        return "FAIL", fact + (
            f"; {plural(len(blocking), 'fail-on message')}, "
            f"first: {p}:{line} {s} {m}")
    if head["code"]:
        return "FAIL", fact + f"; pylint exited {head['code']}"
    if new:
        (s, p, m), line = new[0]
        fact += f"; first new: {p}:{line} {s} {m[:80]}"
    return "PASS", fact


def new_items(head, base):
    def norm(key):
        return (*map(str, key[:-1]), re.sub(r"\d+", "#", str(key[-1])))
    old = collections.Counter(norm(key) for key, _ in base)
    new = []
    for key, line in head:
        if old[norm(key)] > 0:
            old[norm(key)] -= 1
        else:
            new.append([key, line])
    return new


def diffed(ctx, name, measure, show):
    head, base = measured(ctx, "lint", name, measure)
    new = new_items(head, base)
    fact = (f"{plural(len(head), 'finding')}, base "
            f"{len(base)}, {len(new)} new")
    if new:
        return "FAIL", fact + "; first: " + show(new[0])
    return "PASS", fact


def gate_ruff(ctx):
    def measure(root, log):
        code, out = run([ctx.bin / "ruff", "check", "discopy", "-q",
                         "--output-format", "json", "--no-cache"], root, log)
        try:
            found = json.loads(out or "[]")
        except ValueError:
            found = None
        if code not in (0, 1) or not isinstance(found, list):
            raise ToolError(f"ruff exited {code}: {first_line(out)}")
        return [[[f["code"], os.path.relpath(f["filename"], root),
                  f["message"]], f["location"]["row"]] for f in found]
    return diffed(ctx, "ruff", measure,
                  lambda f: f"{f[0][1]}:{f[1]} {f[0][0]} {f[0][2]}")


def gate_ty(ctx):
    pattern = re.compile(r"^(\S+?):(\d+):\d+: (\w+)\[([\w-]+)\] (.*)$")

    def measure(root, log):
        code, out = run([ctx.bin / "ty", "check", "--exit-zero-on-warning",
                         "--output-format", "concise", "--python",
                         ctx.bin.parent], root, log)
        if code not in (0, 1) or "ty failed" in out:
            causes = re.findall(r"Cause: (.*)", out) or [first_line(out)]
            raise ToolError(f"ty exited {code}: {causes[-1]}, "
                            f"{out.strip().splitlines()[-1]}"[:200])
        return [[[m[3], m[4], m[1], m[5]], m[2]] for m in map(
            pattern.match, out.splitlines()) if m]
    return diffed(ctx, "ty", measure,
                  lambda f: f"{f[0][2]}:{f[1]} {f[0][0]}[{f[0][1]}] {f[0][3]}")


def module_tests(root, path):
    parts = pathlib.PurePath(path).parts
    stem = parts[-1][:-3]
    if path in SPECIAL_TESTS:
        candidates = [SPECIAL_TESTS[path]]
    elif len(parts) == 2:
        candidates = [f"test/{stem}.py"]
    elif parts[1] == "optyx":
        sub = "/".join(parts[2:-1])
        names = {f"test_{stem}.py", f"test_{stem.rstrip('s')}.py",
                 f"{stem}.py"}
        candidates = [f"test/optyx/{d}{n}" for n in sorted(names)
                      for d in ({"", f"{sub}/"} if sub else {""})]
    else:
        candidates = [f"test/{parts[1]}/{stem}.py", f"test/{parts[1]}.py"]
    return [c for c in candidates if (root / c).exists()]


def test_files(ctx):
    files, untested, infra = set(), [], []
    for path in ctx.changed:
        if not (ctx.root / path).exists():
            if path.startswith("discopy/") and path.endswith(".py"):
                files.update(module_tests(ctx.root, path))
            continue
        if path.startswith("test/fixtures/") or path == "test/plugin.py" \
                or path.endswith("conftest.py"):
            infra.append(path)
        elif path.startswith("test/") and path.endswith(".py"):
            files.add(path)
        if path.startswith("discopy/") and path.endswith(".py"):
            files.add(path)
            tests = module_tests(ctx.root, path)
            files.update(tests)
            if not tests and not path.endswith("__init__.py"):
                untested.append(path)
    return sorted(files), untested, infra


def failures(out):
    return sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)))


def summary(out):
    lines = [line.strip("= ") for line in out.splitlines()
             if re.search(r"\d+ (passed|failed|errors?|tests? collected)",
                          line)]
    return lines[-1] if lines else first_line(out[-2000:])


def pytest(ctx, root, log, files=(), cov=None, extra=()):
    cmd = [ctx.bin / "python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "-rfE", *extra, *files]
    if cov is not None:
        cmd[1:3] = ["-m", "coverage", "run", f"--data-file={cov}", "-m",
                    "pytest"]
    return run(cmd, root, log)


def gate_tests(ctx):
    if ctx.args.full:
        return gate_full(ctx)
    files, untested, infra = test_files(ctx)
    log = ctx.logs / "tests.log"
    notes = []
    if untested:
        notes.append(f"no test file for {capped(untested, 3)}, doctests only")
    other = sorted({p.split("/")[0] for p in ctx.changed
                    if p.split("/")[0] in ("proptest", "benchmark")})
    if other:
        notes.append(f"{'/, '.join(other)}/ changed, run it (CONTRIBUTING)")
    if infra:
        code, out = pytest(ctx, ctx.root, log, extra=["--collect-only"])
        if code:
            return "FAIL", (f"shared test code changed ({capped(infra, 2)}): "
                            f"collection fails, {last_error(out)}")
        notes.append(f"shared test code changed ({capped(infra, 2)}): "
                     f"{summary(out)}, run --full")
    tail = "".join(f"; {n}" for n in notes)
    if not files:
        return "SKIP", ("no discopy/ or test/ module changed (--full for all)"
                        if not infra else "no module changed" + tail)
    code, out = pytest(ctx, ctx.root, log, files)
    failed = failures(out)
    fact = f"{plural(len(files), 'file')}: {summary(out)}"
    if code not in (0, 5) and not failed:
        return "FAIL", f"pytest exited {code}: {last_error(out)}" + tail
    if "Interrupted:" in out:
        return "FAIL", fact + "; interrupted, not every file ran" + tail
    if not failed:
        return "PASS", fact + tail
    base_failed = set()
    if ctx.key(ctx.tree, "full") == ctx.key(ctx.base, "full"):
        base_failed = set(failed)
    else:
        base = ctx.base_tree()
        old = [f for f in {n.split("::")[0] for n in failed}
               if (base / f).exists()]
        if old:
            base_failed = set(failures(pytest(ctx, base, log, old)[1]))
    new = [n for n in failed if n not in base_failed]
    fact += f"; {len(new)} new, {len(failed) - len(new)} also fail on base"
    return ("FAIL", fact + f"; first: {new[0]}" + tail) if new else (
        "PASS", fact + tail)


def gate_full(ctx):
    log = ctx.logs / "tests.log"

    def measure(root, _log):
        cov = ctx.logs / f"coverage-{'base' if root != ctx.root else 'head'}"
        code, out = pytest(ctx, root, log, cov=cov)
        _, report = run([ctx.bin / "python", "-m", "coverage", "report",
                         f"--data-file={cov}"], root, log)
        total = re.search(r"^TOTAL.*?([\d.]+)%", report, re.M)
        return {"failed": failures(out), "summary": summary(out),
                "coverage": float(total.group(1)) if total else None,
                "code": code, "interrupted": "Interrupted:" in out
                or (code not in (0, 5) and not failures(out))}
    head, base = measured(ctx, "full", "pytest", measure)
    if head["interrupted"]:
        return "FAIL", (f"full suite interrupted, pytest exited "
                        f"{head.get('code')}: {head['summary']}")
    if base["interrupted"]:
        base = {"failed": [], "coverage": None}
    toml = (ctx.root / "pyproject.toml").read_text(encoding="utf-8")
    under = float(re.search(r"^fail_under = ([\d.]+)", toml, re.M).group(1))
    new = [n for n in head["failed"] if n not in set(base["failed"])]
    fact = (f"full suite {head['summary']}; {len(new)} new failures, "
            f"{len(head['failed']) - len(new)} also on base; coverage "
            f"{head['coverage']}% (base {base['coverage']}%, "
            f"fail_under {under})")
    if new:
        return "FAIL", fact + f"; first: {new[0]}"
    if head["coverage"] is None or head["coverage"] < under:
        return "FAIL", fact + "; coverage below fail_under"
    return "PASS", fact


def gate_notebooks(ctx):
    names = [] if ctx.args.all_notebooks else sorted(
        p[len("docs/notebooks/"):-3] for p in ctx.under("docs/notebooks/")
        if p.endswith(".md") and (ctx.root / p).exists())
    if not names and not ctx.args.all_notebooks:
        return "SKIP", "no notebook changed (--all-notebooks for all)"
    code, out = run([ctx.bin / "python", "docs/export_notebooks.py", "--check",
                     "--jobs", "2", *names], ctx.root,
                    ctx.logs / "notebooks.log")
    what = ", ".join(names) or "all"
    if code == 0:
        return "PASS", what
    failed = re.findall(r"failed to execute: (.*)", out)
    return "FAIL", "failed to execute: " + (failed[-1] if failed else what)


def gate_changelog(ctx):
    path = ctx.root / "CHANGELOG.md"
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    conflict = [i + 1 for i, line in enumerate(lines)
                if re.match(r"(<{7}|={7}|>{7})( |$)", line)]
    if conflict:
        return "FAIL", f"conflict marker at CHANGELOG.md:{conflict[0]}"
    if b"\r\n" in raw and b"\r\n" not in subprocess.run(
            ["git", "show", f"{ctx.base}:CHANGELOG.md"], cwd=ctx.root,
            capture_output=True, check=False).stdout:
        return "FAIL", "CRLF line endings, base has LF"
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("## [Unreleased]")), None)
    if start is None:
        return "FAIL", "no '## [Unreleased]' heading"
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## [")), len(lines))
    added = []
    diff = git("diff", "-U0", ctx.base, "--", "CHANGELOG.md", cwd=ctx.root)
    for hunk in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff,
                            re.M):
        first, count = int(hunk.group(1)), int(hunk.group(2) or 1)
        added += [i for i in range(first - 1, first - 1 + count)
                  if start < i < end and lines[i].strip()]
    removed = [line[1:] for line in diff.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    bullets = [i for i in added if lines[i].startswith("- ")]
    if not added:
        if ctx.under("discopy/"):
            return "SKIP", ("no bullet added; discopy/ changed, add one if "
                            "user-facing")
        return "SKIP", "no bullet added"
    text = " ".join(re.sub(r"`[^`]*`", "", lines[i]) for i in added)
    refs = set(re.findall(r"\[([^\]\[]+)\](?![(:\[])", text))
    defined = {m.group(1) for m in map(
        re.compile(r"^\[([^\]]+)\]: \S").match, lines) if m}
    missing = sorted(refs - defined)
    what = plural(len(bullets), "bullet") if bullets else (
        plural(len(added), "line") + " changed")
    fact = f"{what}, refs {capped(sorted(refs)) or 'none'}"
    if removed:
        fact += f"; also removes {plural(len(removed), 'line')} of base's, " \
            f"first: {removed[0][:60]}"
    if missing:
        return "FAIL", fact + "; no link definition for " + capped(
            f"[{m}]" for m in missing)
    return ("PASS" if bullets else "SKIP"), fact


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="comma-separated: " + ",".join(GATES))
    parser.add_argument("--full", action="store_true",
                        help="the whole suite with coverage, diffed with base")
    parser.add_argument("--all-notebooks", action="store_true")
    parser.add_argument("--base", default="origin/main",
                        help="compare with merge-base(HEAD, BASE)")
    args = parser.parse_args()
    gates = args.only.split(",") if args.only else list(GATES)
    unknown = set(gates) - set(GATES)
    if unknown:
        parser.error(f"unknown gate(s): {', '.join(sorted(unknown))}")
    begin = time.time()
    try:
        ctx = Context(args)
    except subprocess.CalledProcessError as error:
        print(f"FAIL setup: {' '.join(error.cmd)}: "
              f"{first_line(error.stderr or '')}")
        return 1
    failed = ctx.lock_error is not None
    if failed:
        print(f"FAIL lock: {ctx.lock_error} (run uv lock); gates below use "
              f"{ctx.bin.parent}")
    order = sorted(gates, key=lambda g: g != "pylint")
    try:
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = {g: pool.submit(globals()[f"gate_{g}"], ctx)
                       for g in order}
            for gate in gates:
                try:
                    status, fact = futures[gate].result()
                except ToolError as error:
                    status, fact = "FAIL", str(error)
                except Exception as error:
                    status, fact = "FAIL", f"crashed: {error!r}"[:200]
                failed |= status == "FAIL"
                print(f"{status} {gate}: {fact}", flush=True)
    finally:
        ctx.close()
    print(f"base {ctx.base[:10]}, {int(time.time() - begin)}s, "
          f"logs {ctx.logs}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
