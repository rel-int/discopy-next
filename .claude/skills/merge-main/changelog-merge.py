"""Merge driver for CHANGELOG.md: main's file plus the branch's own changes.

Usage, as git calls it: ``changelog-merge.py %O %A %B``, writing the result to
%A. When the branch (%A, ours) merges main (%B, theirs), the result is main's
file with the branch's own changes since the base %O applied to it: its new
bullets in the section where it put them, its new link definitions in number
order among their kind, its rewordings and deletions of base lines. A line
main deleted is never brought back. A bullet with its two-space continuation
lines is one unit. A clean line merge is git's own. A base bullet both sides
edited differently, or that one side deleted and the other reworded, is
written between conflict markers, and the exit status is 1. A release on
main (a ``## `` heading added, renamed or deleted) leaves the whole file to
git's markers. Line endings and undecodable bytes are kept.

After changing this driver, re-run ``replay.py`` beside it.
"""
import difflib
import re
import subprocess
import sys

LINK = re.compile(r"^\[([^\]#\s]*#)(\d+)\]:")
SIMILAR = 0.6


def units(text):
    """Split a changelog into bullets with their continuations and lines."""
    out = []
    for line in text.splitlines(keepends=True):
        if out and out[-1].startswith("- ") and line.startswith("  ") \
                and line.strip():
            out[-1] += line
        else:
            out.append(line)
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    return out


def kind(unit):
    """The kind of a unit, the key of its pairing with a rewording."""
    if match := LINK.match(unit):
        return ("link", match.group(1), match.group(2))
    if unit.startswith("- "):
        return ("bullet",)
    return ("line", unit)


def link_key(unit):
    """The prefix and number of a link definition, or None."""
    match = unit and LINK.match(unit)
    return match and (match.group(1), int(match.group(2)))


def similarity(one, two):
    """The ratio of words two bullets share, in order."""
    matcher = difflib.SequenceMatcher(None, one.split(), two.split(),
                                      autojunk=False)
    if matcher.real_quick_ratio() < SIMILAR or matcher.quick_ratio() \
            < SIMILAR:
        return 0
    return matcher.ratio()


def align(base, other):
    """Map base indices to the other's, and pair the rewordings of the rest."""
    kept, reworded, added = {}, {}, []
    matcher = difflib.SequenceMatcher(None, base, other, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            kept.update(zip(range(i1, i2), range(j1, j2)))
            continue
        free = list(range(j1, j2))
        for i in range(i1, i2):
            if kind(base[i])[0] == "line":
                continue
            scores = [(1 if kind(base[i])[0] == "link"
                       else similarity(base[i], other[j]), j)
                      for j in free if kind(base[i]) == kind(other[j])]
            score, j = max(scores, default=(0, None))
            if score >= SIMILAR:
                reworded[i] = j
                free.remove(j)
        added.extend(free)
    return kept, reworded, sorted(added)


class Slot:
    """One unit of the result, remembering where it came from."""

    def __init__(self, text, side, conflict=None):
        self.text, self.side, self.conflict = text, side, conflict


def conflict_text(ours, theirs):
    """Standard conflict markers around the two versions of one unit."""
    return ("<<<<<<< ours\n" + (ours or "") + "=======\n" + (theirs or "")
            + ">>>>>>> theirs\n")


def place_link(result, slot):
    """Insert a link definition in number order among those of its kind."""
    prefix, number = link_key(slot.text)
    same = [k for k, other in enumerate(result)
            if (key := link_key(other.text)) and key[0] == prefix]
    if not same:
        return False
    after = [k for k in same if link_key(result[k].text)[1] < number]
    result.insert(after[-1] + 1 if after else same[0], slot)
    return True


def in_order(result, k):
    """Whether the link at k sits between smaller and larger of its kind."""
    prefix, number = link_key(result[k].text)
    near = [link_key(result[j].text) for j in (k - 1, k + 1)
            if 0 <= j < len(result)]
    return all(key is None or key[0] != prefix
               or (key[1] < number) == (j == 0)
               for j, key in enumerate(near))


def resolve(base, ours, theirs):
    """The merged units and the number of conflicts."""
    o_kept, o_reworded, o_added = align(base, ours)
    t_kept, t_reworded, _ = align(base, theirs)
    result = [Slot(text, "theirs") for text in theirs]
    at_ours, pending, conflicts = {}, {}, 0
    for i in range(len(base)):
        if i in t_kept:
            slot = result[t_kept[i]]
            if i in o_kept:
                at_ours[o_kept[i]] = slot
            elif i in o_reworded:
                slot.text, slot.side = ours[o_reworded[i]], "ours"
                at_ours[o_reworded[i]] = slot
            else:
                slot.text = None
            continue
        if i in o_kept:
            continue
        mine = ours[o_reworded[i]] if i in o_reworded else None
        main = theirs[t_reworded[i]] if i in t_reworded else None
        if mine == main:
            if mine is not None:
                at_ours[o_reworded[i]] = result[t_reworded[i]]
            continue
        conflicts += 1
        if main is None:
            pending[o_reworded[i]] = Slot(None, "ours", (mine, None))
            continue
        result[t_reworded[i]].conflict = (mine, main)
        if mine is not None:
            at_ours[o_reworded[i]] = result[t_reworded[i]]
    base_only = {text for text in set(base) - set(theirs) if text.strip()}
    fresh = {text: slot for slot in result if slot.side == "theirs"
             and (text := slot.text) and text not in base and text.strip()}
    for j in sorted(o_added + list(pending)):
        slot = pending.get(j) or Slot(ours[j], "ours")
        if j not in pending and ours[j] in base_only:
            continue
        if j not in pending and ours[j] in fresh:
            at_ours[j] = fresh[ours[j]]
            continue
        blank = None if j in pending else shared_blank(result, ours, j,
                                                       at_ours, fresh)
        if blank:
            at_ours[j] = blank
            continue
        at_ours[j] = insert(result, slot, at_ours, j, fresh)
    return result, conflicts


def shared_blank(result, ours, j, at_ours, fresh):
    """Main's blank line beside a heading both sides added, for ours[j]."""
    if ours[j].strip():
        return None
    before = at_ours.get(j - 1)
    after = fresh.get(ours[j + 1]) if j + 1 < len(ours) else None
    for slot, step in ((before, 1), (after, -1)):
        if slot in fresh.values() and slot in result:
            k = result.index(slot) + step
            if 0 <= k < len(result) and result[k].text is not None \
                    and not result[k].text.strip():
                return result[k]
    return None


def insert(result, slot, at_ours, j, fresh):
    """Put an ours unit after the nearest ours unit already placed."""
    anchor = next((at_ours[k] for k in range(j - 1, -1, -1)
                   if k in at_ours and at_ours[k] in result), None)
    pos = result.index(anchor) + 1 if anchor else 0

    def contested(k):
        return k < len(result) and result[k].text in fresh \
            and result[k].side == "theirs"
    if slot.conflict is None and link_key(slot.text):
        if not contested(pos):
            result.insert(pos, slot)
            if in_order(result, pos):
                return slot
            result.remove(slot)
        if not place_link(result, slot):
            result.insert(end_of_block(result, pos), slot)
        return slot
    result.insert(pos, slot)
    return slot


def end_of_block(result, pos):
    """Past the run of link definitions of one kind that pos splits."""
    while 0 < pos < len(result) and (key := link_key(result[pos].text)) \
            and (prev := link_key(result[pos - 1].text)) \
            and key[0] == prev[0]:
        pos += 1
    return pos


def render(result):
    """The text of the merged units, conflicts between markers."""
    return "".join(conflict_text(*slot.conflict) if slot.conflict
                   else slot.text for slot in result
                   if slot.conflict or slot.text is not None)


def headings(text):
    """The version headings of a changelog, in order."""
    return [line for line in text.splitlines() if line.startswith("## ")]


def merge(base, ours, theirs):
    """The merged text and the number of conflicts, or None for a release.

    When main added, renamed or deleted a ``## `` heading, as a release does,
    a bullet the branch added could land under the released version, so the
    branch's changes are left to git's conflict markers.
    """
    if headings(base) != headings(theirs) and units(ours) != units(base):
        return None
    result, conflicts = resolve(units(base), units(ours), units(theirs))
    merged = render(result)
    if theirs and not theirs.endswith("\n") and merged.endswith("\n"):
        merged = merged[:-1]
    return merged, conflicts


def git_merge(paths):
    """Git's own line merge of the three files, or None on a conflict."""
    proc = subprocess.run(["git", "merge-file", "-p", *paths],
                          capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def read(path):
    """The text of a file, its line endings and undecodable bytes kept."""
    with open(path, encoding="utf-8", errors="surrogateescape",
              newline="") as file:
        return file.read()


def main(base_path, ours_path, theirs_path):
    """Merge the three files into ours_path, exiting 1 on a conflict.

    Any failure of the unit merge, a crash included, falls back to git's
    conflict markers over the whole file, so a conflict always shows.
    """
    clean = git_merge([ours_path, base_path, theirs_path])
    if clean is not None:
        with open(ours_path, "wb") as file:
            file.write(clean)
        return 0
    try:
        merged = merge(*map(read, (base_path, ours_path, theirs_path)))
        if merged is not None:
            with open(ours_path, "w", encoding="utf-8",
                      errors="surrogateescape", newline="") as file:
                file.write(merged[0])
            return 1 if merged[1] else 0
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    subprocess.run(["git", "merge-file", "-L", "ours", "-L", "base",
                    "-L", "theirs", ours_path, base_path, theirs_path],
                   check=False)
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
