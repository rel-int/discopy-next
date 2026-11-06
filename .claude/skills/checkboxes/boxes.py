"""Start, claim, tick, extend and finish the TODO.md of a branch, one pushed commit per step.

RULES.md 1-3: the prompt is quoted verbatim at the top and never altered, a box is claimed as
`[WIP] @<session>-<yyyy-MM-dd HH:mm>` (UTC, stale after 12 h) in a one-line commit pushed before
work starts, and TODO.md is deleted once every box is `[x]` (filed ones too). Every step pushes
to the branch of the same name on the remote, never from main. A push rejected as
non-fast-forward is retried once after merging the remote branch (RULES.md 4, never a rebase),
re-checking the box; any other failure undoes the step's commit.

Usage:
  boxes.py start (--prompt TEXT | --from-file FILE) [--box TEXT ...] [--note TEXT] [--round]
                [--title TEXT]
  boxes.py list
  boxes.py claim N [--force]
  boxes.py done N [--filed ISSUE] [--force]
  boxes.py add TEXT [--claim]
  boxes.py finish [--no-changelog] [--base REF] [--message TEXT]
Common: --dry-run, --session ID, --trailer LINE (repeatable).
Exit codes: 0 done, 1 refused or failed, 2 the box was claimed by someone else meanwhile.
"""

import argparse
import collections
import datetime
import difflib
import os
import re
import subprocess
import sys

TODO = "TODO.md"
STALE = datetime.timedelta(hours=12)
BOX = re.compile(r"^(\s*)- \[( |x|X|WIP)\] ?(.*?)(\r?\n?)$")
STAMP = re.compile(
    r"^@(\S+?)(?:-(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?Z?)?(?=\s|$) ?(.*)$"
)
FILED = re.compile(r", filed as (?:[\w./-]*)#\d+\s*$")
BULLET = re.compile(r"^[-*] ")
REF = re.compile(r"\[((?:[\w.-]+)?#\d+)\](?![(:])")
DEF = re.compile(r"^\[(([\w.-]*)#(\d+))\]:\s", re.M)
REJECTED = re.compile(
    r"^ ! \[rejected\].*\((?:fetch first|non-fast-forward)\)", re.M
)
PROTECTED = {"main", "master"}


class Refused(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def git(*args, check=True, text_input=None):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        input=text_input,
        check=False,
    )
    if check and result.returncode:
        raise Refused(
            f"git {' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(
        second=0, microsecond=0
    )


def session_id(args):
    sid = args.session or os.environ.get("CLAUDE_CODE_REMOTE_SESSION_ID", "")
    if sid.startswith("cse_"):
        sid = "session_" + sid[len("cse_") :]
    if not sid:
        raise Refused("no session id: pass --session ID")
    return sid


def claimed_at(date, time):
    if not date:
        return None
    try:
        when = datetime.datetime.strptime(
            f"{date} {time or '23:59'}", "%Y-%m-%d %H:%M"
        )
    except ValueError:
        return None
    return when.replace(tzinfo=datetime.timezone.utc)


class Todo:
    def __init__(self, text):
        self.lines = text.splitlines(keepends=True)
        self.boxes = []
        for i, line in enumerate(self.lines):
            if BOX.match(line):
                self.boxes.append([i, i + 1])
            elif (
                self.boxes
                and self.boxes[-1][1] == i
                and line.strip()
                and len(line) - len(line.lstrip())
                > len(BOX.match(self.lines[self.boxes[-1][0]]).group(1))
            ):
                self.boxes[-1][1] = i + 1
        self.header = "".join(
            self.lines[: self.boxes[0][0]] if self.boxes else self.lines
        )
        ends = [line[len(line.rstrip("\r\n")) :] for line in self.lines]
        self.eol = max(set(ends) - {""}, key=ends.count, default="\n")

    def text(self):
        return "".join(self.lines)

    def box(self, n):
        if not 1 <= n <= len(self.boxes):
            raise Refused(f"no box {n}: TODO.md has {len(self.boxes)}")
        start, end = self.boxes[n - 1]
        indent, state, rest, eol = BOX.match(self.lines[start]).groups()
        claim = STAMP.match(rest) if state == "WIP" else None
        return {
            "line": start,
            "indent": indent,
            "state": state.lower(),
            "body": claim.group(4) if claim else rest,
            "eol": eol or self.eol,
            "owner": claim.group(1) if claim else None,
            "since": claimed_at(claim.group(2), claim.group(3))
            if claim
            else None,
            "full": "".join(self.lines[start:end]),
        }

    def settled(self, n):
        return self.box(n)["state"] == "x"

    def filed(self, n):
        return self.settled(n) and bool(FILED.search(self.box(n)["full"]))

    def set_box(self, n, marker, body):
        b = self.box(n)
        self.lines[b["line"]] = f"{b['indent']}- {marker} {body}{b['eol']}"

    def append_to_box(self, n, suffix):
        last = self.boxes[n - 1][1] - 1
        line = self.lines[last]
        body = line.rstrip("\r\n")
        self.lines[last] = body + suffix + (line[len(body) :] or self.eol)


def edited(text, todo):
    old, new = Todo(text).header, Todo(todo.text()).header
    if new.rstrip("\r\n") != old.rstrip("\r\n"):
        raise Refused(
            "this step would alter the text above the first box (RULES.md 1)"
        )
    return todo.text()


def age_of(b):
    return None if b["since"] is None else now() - b["since"]


def hours(age):
    return "?" if age is None else f"{age.days * 24 + age.seconds // 3600}h"


def short(text, width=60):
    text = re.sub(r"\s+", " ", text.replace("**", "")).strip().rstrip(":;,.")
    if len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0].rstrip(":;,.") + "…"


def quote(prompt):
    lines = prompt.rstrip("\n").split("\n")
    if all(not line.strip() or line.startswith(">") for line in lines):
        return "\n".join(lines) + "\n"
    return (
        "\n".join("> " + line if line.strip() else ">" for line in lines)
        + "\n"
    )


def stamp(sid):
    return f"[WIP] @{sid}-{now():%Y-%m-%d %H:%M}"


def held_by_other(b, sid, force):
    if b["state"] != "wip" or b["owner"] == sid or force:
        return None
    age = age_of(b)
    if age is None:
        return f"claimed by @{b['owner']} at an unknown time: --force to take it over"
    if age < STALE:
        return (
            f"claimed by @{b['owner']} since {b['since']:%Y-%m-%d %H:%M}"
            f" ({hours(age)} ago, stale after 12h): pick another box"
        )
    return None


def read_todo(required=True):
    if not os.path.exists(TODO):
        if required:
            raise Refused("no TODO.md here: start one with `boxes.py start`")
        return None
    if git("status", "--porcelain", "--", TODO).stdout.strip():
        raise Refused(
            "TODO.md has uncommitted changes: commit or restore them first"
        )
    with open(TODO, encoding="utf-8", newline="") as f:
        return f.read()


def changelog_bullets(text):
    match = re.search(
        r"^## \[Unreleased\]\r?\n(.*?)(?=^## \[)", text, re.M | re.S
    )
    bullets = []
    for line in (match.group(1) if match else "").splitlines():
        if BULLET.match(line):
            bullets.append(line)
        elif bullets and line.startswith(" ") and line.strip():
            bullets[-1] += "\n" + line
    return bullets


def show(ref, path):
    result = git("show", f"{ref}:{path}", check=False)
    return result.stdout if result.returncode == 0 else ""


def out_of_order(prev, number, after):
    if prev is not None and after is not None and prev > after:
        return not (prev < number or number < after)
    return (prev is not None and prev > number) or (
        after is not None and number > after
    )


def check_changelog(base, need_bullet):
    if git(
        "rev-parse", "--verify", "-q", f"{base}^{{commit}}", check=False
    ).returncode:
        raise Refused(
            f"--base {base} is not a commit: fetch it or pass --base REF"
        )
    mb = git("merge-base", "HEAD", base, check=False).stdout.strip()
    if not mb:
        raise Refused(f"HEAD and {base} have no merge base")
    head = show("HEAD", "CHANGELOG.md")
    olds = [show(mb, "CHANGELOG.md"), show(base, "CHANGELOG.md")]
    old_bullets = {b for old in olds for b in changelog_bullets(old)}
    added = [b for b in changelog_bullets(head) if b not in old_bullets]
    problems, warnings = [], []
    if not added:
        message = (
            f"CHANGELOG.md adds no bullet under ## [Unreleased] since {mb[:8]}"
        )
        (problems if need_bullet else warnings).append(
            message
            + ("" if need_bullet else " (--no-changelog: not user-facing)")
        )
    defs = DEF.findall(head)
    names = {d[0] for d in defs}
    for bullet in added:
        for ref in REF.findall(bullet):
            if ref not in names:
                problems.append(
                    f"[{ref}] has no link definition (e.g. `[{ref}]: https://…`)"
                )
    old_counts = [collections.Counter(d[0] for d in DEF.findall(old))
                  for old in olds]
    old_names = {name for counts in old_counts for name in counts}
    counts = collections.Counter(d[0] for d in defs)
    for name, count in counts.items():
        if count > max(1, *(c[name] for c in old_counts)):
            problems.append(f"[{name}] is defined twice")
    by_kind = {}
    for name, kind, number in defs:
        by_kind.setdefault(kind, []).append((int(number), name))
    for kind, items in by_kind.items():
        numbers = [n for n, _ in items]
        for i, (number, name) in enumerate(items):
            prev = numbers[i - 1] if i else None
            after = numbers[i + 1] if i + 1 < len(items) else None
            if name not in old_names and out_of_order(prev, number, after):
                problems.append(
                    f"link [{name}] is out of number order among the [{kind}#N]"
                    " definitions"
                )
    if git("status", "--porcelain", "--", "CHANGELOG.md").stdout.strip():
        warnings.append(
            "CHANGELOG.md has uncommitted changes: only HEAD's file was checked"
        )
    return added, problems, warnings


def trailers(args):
    configured = git(
        "config", "--get-all", "next.trailer", check=False
    ).stdout.splitlines()
    return [t for t in configured + (args.trailer or []) if t.strip()]


def target(base):
    branch = git(
        "symbolic-ref", "--short", "-q", "HEAD", check=False
    ).stdout.strip()
    if not branch:
        raise Refused("detached HEAD: check out the PR branch")
    if branch in PROTECTED | {base.rpartition("/")[2]}:
        raise Refused(
            f"on {branch}: TODO.md lives on a PR branch, never on main (STYLE.md)"
        )
    upstream = git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{u}",
        check=False,
    ).stdout.strip()
    remote, _, rbranch = upstream.partition("/")
    if not upstream or rbranch != branch:
        remote = "origin"
    return branch, remote


def restore(had_todo):
    if had_todo:
        git("reset", "-q", "--", TODO, check=False)
        git("checkout", "-q", "HEAD", "--", TODO, check=False)
    else:
        git(
            "rm", "-q", "--cached", "--ignore-unmatch", "--", TODO, check=False
        )
        if os.path.exists(TODO):
            os.remove(TODO)


def commit(message, new_text, had_todo):
    if new_text is None:
        git("rm", "-q", "--", TODO)
    else:
        with open(TODO, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        git("add", "--", TODO)
    result = git(
        "commit",
        "-F",
        "-",
        "--only",
        "--",
        TODO,
        text_input=message,
        check=False,
    )
    if result.returncode:
        restore(had_todo)
        raise Refused(
            "git commit failed (a hook, signing or identity?), TODO.md restored: "
            + (result.stderr.strip() or result.stdout.strip() or "no output")
        )
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def undo(had_todo):
    git("reset", "-q", "--soft", "HEAD~1")
    restore(had_todo)


def push(branch, remote):
    result = git(
        "push", "-q", "-u", remote, f"HEAD:refs/heads/{branch}", check=False
    )
    if result.returncode == 0:
        return None
    return result.stderr.strip() or "no output"


def merge_remote(branch, remote, message_trailers):
    git("fetch", "-q", remote, branch)
    ref = f"{remote}/{branch}"
    message = f"Merge remote-tracking branch '{ref}' into {branch}\n"
    if message_trailers:
        message += "\n" + "\n".join(message_trailers) + "\n"
    result = git("merge", "-q", "--no-edit", "-m", message, ref, check=False)
    if result.returncode:
        conflicted = (
            git(
                "rev-parse", "-q", "--verify", "MERGE_HEAD", check=False
            ).returncode
            == 0
        )
        if conflicted:
            git("merge", "--abort", check=False)
        raise Refused(
            f"nothing committed; merging {ref} "
            + (
                "conflicts: merge it by hand (RULES.md 4), then retry"
                if conflicted
                else "did not start: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        )


def run(step, args):
    text = read_todo(required=args.command != "start")
    subject, new_text = step(text)
    tr = trailers(args)
    message = subject + "\n" + ("\n" + "\n".join(tr) + "\n" if tr else "")
    if args.dry_run:
        print(
            "dry run, would commit:\n"
            + "\n".join("  " + m for m in message.splitlines())
        )
        diff = difflib.unified_diff(
            (text or "").splitlines(),
            (new_text or "").splitlines(),
            "a/" + TODO,
            "b/" + TODO if new_text is not None else "/dev/null",
            lineterm="",
            n=0,
        )
        print("\n".join(list(diff)[2:]))
        return 0
    branch, remote = target(getattr(args, "base", "origin/main"))
    sha = commit(message, new_text, text is not None)
    error = push(branch, remote)
    if error is None:
        print(f"boxes: {sha} {subject}; pushed")
        return 0
    undo(text is not None)
    if not REJECTED.search(error):
        raise Refused(
            f"push to {remote} {branch} failed, commit undone: {error}"
        )
    print(
        "boxes: push rejected (non-fast-forward): merging the remote branch and retrying once",
        flush=True,
    )
    merge_remote(branch, remote, tr)
    text = read_todo(required=args.command != "start")
    if args.command == "start" and text is not None:
        raise Refused(
            "someone started a TODO.md meanwhile: read it, then `boxes.py add`",
            2,
        )
    subject, new_text = step(text)
    message = subject + "\n" + ("\n" + "\n".join(tr) + "\n" if tr else "")
    sha = commit(message, new_text, text is not None)
    error = push(branch, remote)
    if error is not None:
        undo(text is not None)
        raise Refused(
            f"push failed again, commit undone (the merge stays): {error}"
        )
    print(f"boxes: {sha} {subject}; pushed after merging the remote branch")
    return 0


def cmd_start(args):
    if args.from_file:
        try:
            with open(args.from_file, encoding="utf-8") as f:
                source = f.read()
        except OSError as error:
            raise Refused(f"--from-file: {error}") from error
    else:
        source = args.prompt
    lines = source.splitlines(keepends=True)
    first = next(
        (i for i, line in enumerate(lines) if BOX.match(line)), len(lines)
    )
    prompt = "".join(lines[:first])
    file_boxes = [line for line in lines[first:] if line.strip()]
    parsed = Todo("".join(file_boxes))
    covered = {i for start, end in parsed.boxes for i in range(start, end)}
    stray = [line for i, line in enumerate(parsed.lines) if i not in covered]
    if stray:
        raise Refused(
            "--from-file: not a box or its indented continuation: "
            + short(stray[0])
        )
    if not prompt.strip().strip(">").strip():
        raise Refused(
            "the prompt is empty: RULES.md 1 starts TODO.md with it verbatim"
        )
    if os.path.exists(TODO):
        if args.round:
            raise Refused(
                "TODO.md exists, so this round is still open: add its points with"
                " `boxes.py add`"
            )
        raise Refused(
            "TODO.md exists: finish it first (RULES.md 2), or `boxes.py add`"
        )
    boxes = [line.rstrip("\r\n") + "\n" for line in file_boxes]
    boxes += [f"- [ ] {b}\n" for b in args.box or []]
    if not boxes:
        raise Refused("no boxes: pass --box TEXT for each point")

    def step(_):
        parts = [quote(prompt)]
        if args.note:
            parts.append(args.note.rstrip("\n") + "\n")
        parts.append("".join(boxes))
        title = args.title or short(BOX.match(boxes[0]).group(3))
        subject = (
            "Start TODO.md for the review round"
            if args.round
            else "Start TODO.md"
        )
        return f"{subject}: {title}", "\n".join(parts)

    return run(step, args)


def cmd_list(_):
    todo = Todo(read_todo())
    if not todo.boxes:
        print("TODO.md has no boxes")
    for n in range(1, len(todo.boxes) + 1):
        b = todo.box(n)
        if b["state"] == "wip":
            age = age_of(b)
            state = f"WIP @{b['owner']} {hours(age)}" + (
                " stale" if age is not None and age >= STALE else ""
            )
        elif b["state"] == "x":
            state = "filed" if todo.filed(n) else "x"
        else:
            state = " "
        print(f"{n} [{state}] {short(b['body'], 90)}")
    return 0


def cmd_claim(args):
    sid = session_id(args)

    def step(text):
        todo = Todo(text)
        b = todo.box(args.n)
        if b["state"] == "x":
            raise Refused(f"box {args.n} is already [x]: {short(b['body'])}")
        if b["state"] == "wip" and b["owner"] == sid:
            raise Refused(
                f"box {args.n} is already claimed by this session", 0
            )
        other = held_by_other(b, sid, args.force)
        if other:
            raise Refused(f"box {args.n} is {other}", 2)
        todo.set_box(args.n, stamp(sid), b["body"])
        return f"Claim {short(b['body'])} in TODO.md", edited(text, todo)

    return run(step, args)


def cmd_done(args):
    try:
        sid = session_id(args)
    except Refused:
        sid = None
    filed = None
    if args.filed:
        filed = args.filed if "#" in args.filed else "#" + args.filed
        if not FILED.search(", filed as " + filed):
            raise Refused(f"--filed {args.filed}: not an issue, e.g. 471")

    def step(text):
        todo = Todo(text)
        b = todo.box(args.n)
        if b["state"] == "x":
            raise Refused(f"box {args.n} is already [x]")
        if b["state"] == " " and not (filed or args.force):
            raise Refused(
                f"box {args.n} is unclaimed: claim it first, or --force"
            )
        other = held_by_other(b, sid, args.force)
        if other:
            raise Refused(
                f"box {args.n} is {other}; --force to tick it anyway", 2
            )
        todo.set_box(args.n, "[x]", b["body"])
        if filed:
            todo.append_to_box(args.n, f", filed as {filed}")
        verb = (
            f"File {short(b['body'], 50)} as {filed}"
            if filed
            else f"Tick {short(b['body'])}"
        )
        return f"{verb} in TODO.md", edited(text, todo)

    return run(step, args)


def cmd_add(args):
    sid = session_id(args) if args.claim else None

    def step(text):
        todo = Todo(text)
        marker = stamp(sid) if sid else "[ ]"
        lines = todo.lines
        at = todo.boxes[-1][1] if todo.boxes else len(lines)
        if at and not lines[at - 1].endswith("\n"):
            lines[at - 1] += todo.eol
        if not todo.boxes and at and lines[at - 1].strip():
            lines.insert(at, todo.eol)
            at += 1
        lines.insert(at, f"- {marker} {args.text}{todo.eol}")
        return f"Add {short(args.text)} to TODO.md", edited(text, todo)

    return run(step, args)


def cmd_finish(args):
    def step(text):
        todo = Todo(text)
        if not todo.boxes:
            raise Refused(
                "TODO.md has no boxes: nothing says the work is done"
            )
        open_boxes = [
            n for n in range(1, len(todo.boxes) + 1) if not todo.settled(n)
        ]
        if open_boxes:
            raise Refused(
                "not every box is [x]: "
                + "; ".join(
                    f"{n} {short(todo.box(n)['body'], 50)}" for n in open_boxes
                )
            )
        added, problems, warnings = check_changelog(
            args.base, not args.no_changelog
        )
        for w in warnings:
            print("warning: " + w)
        if problems:
            raise Refused(
                "CHANGELOG.md:\n" + "\n".join("  " + p for p in problems)
            )
        print(
            f"changelog: {len(added)} new bullet(s), every link defined and in order"
        )
        filed = any(todo.filed(n) for n in range(1, len(todo.boxes) + 1))
        subject = args.message or (
            "Delete TODO.md: every point is [x]"
            + (" or filed" if filed else "")
        )
        return subject, None

    return run(step, args)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--session")
    common.add_argument("--trailer", action="append")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", parents=[common])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--from-file")
    p.add_argument("--box", action="append")
    p.add_argument("--note")
    p.add_argument("--title")
    p.add_argument("--round", action="store_true")
    sub.add_parser("list", parents=[common])
    p = sub.add_parser("claim", parents=[common])
    p.add_argument("n", type=int)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("done", parents=[common])
    p.add_argument("n", type=int)
    p.add_argument("--filed")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("add", parents=[common])
    p.add_argument("text")
    p.add_argument("--claim", action="store_true")
    p = sub.add_parser("finish", parents=[common])
    p.add_argument("--no-changelog", action="store_true")
    p.add_argument("--base", default="origin/main")
    p.add_argument("--message")
    args = parser.parse_args(argv)
    top = git("rev-parse", "--show-toplevel", check=False).stdout.strip()
    if not top:
        print("boxes: not in a git repository", file=sys.stderr)
        return 1
    os.chdir(top)
    try:
        return globals()["cmd_" + args.command](args)
    except Refused as refusal:
        print(f"boxes: {refusal}", file=sys.stderr)
        return refusal.code


if __name__ == "__main__":
    sys.exit(main())
