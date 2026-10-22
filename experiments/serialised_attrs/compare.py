"""Read another tree's corpus in the discopy at argv[1]:
compare.py TREE THEIRS.json OUT.json"""
import json, pickle, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
import discopy
assert discopy.__file__.startswith(sys.argv[1]), discopy.__file__
from discopy.utils import dumps, loads

def attempt(f):
    try:
        return f()
    except BaseException as e:  # noqa
        return Exception(type(e).__name__ + ": " + str(e)[:150])

def same(a, b):
    try:
        return type(a) is type(b) and bool(a == b)
    except BaseException:
        return False

theirs = json.load(open(sys.argv[2]))
rows = []
for x in theirs:
    row = {"type": x["type"], "repr_theirs": x["repr"]}
    if x["pickle"].startswith("!"):
        row["status"] = "untransportable"
        rows.append(row); continue
    obj = attempt(lambda: pickle.loads(bytes.fromhex(x["pickle"])))
    if isinstance(obj, Exception):
        row["status"] = "pickle-load-fails"; row["error"] = str(obj)
        rows.append(row); continue
    row["status"] = "ok"
    mine_repr = attempt(lambda: repr(obj))
    mine_tree = attempt(lambda: dumps(obj))
    row["repr_mine"] = mine_repr if isinstance(mine_repr, str) else "!" + str(mine_repr)
    row["tree_mine"] = mine_tree if isinstance(mine_tree, str) else "!" + str(mine_tree)
    row["tree_theirs"] = x["tree"]
    row["repr_same"] = row["repr_mine"] == x["repr"]
    row["tree_same"] = row["tree_mine"] == x["tree"]
    import sys as _sys
    env = dict(vars(discopy))
    env.update(vars(_sys.modules[type(obj).__module__]))
    old = attempt(lambda: eval(x["repr"], env))
    row["their_repr_evals"] = (not isinstance(old, Exception)) and same(old, obj)
    if not x["tree"].startswith("!"):
        back = attempt(lambda: loads(x["tree"]))
        row["their_tree_reads"] = (not isinstance(back, Exception)) and same(back, obj)
        if isinstance(back, Exception):
            row["their_tree_error"] = str(back)
    rows.append(row)
json.dump(rows, open(sys.argv[3], "w"))
from collections import Counter
print(Counter(r["status"] for r in rows))
print("their repr evaluates here to an equal object:",
      sum(r.get("their_repr_evals") is True for r in rows), "of",
      sum("their_repr_evals" in r for r in rows))
ok = [r for r in rows if r["status"] == "ok"]
print("repr differs:", sum(not r["repr_same"] for r in ok),
      " tree differs:", sum(not r["tree_same"] for r in ok),
      " their tree unreadable or unequal:",
      sum(r.get("their_tree_reads") is False for r in ok), "of",
      sum("their_tree_reads" in r for r in ok))
