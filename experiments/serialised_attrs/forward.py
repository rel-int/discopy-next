"""forward.py TREE ROWS.json: in the discopy at TREE, read the trees another
tree wrote (rows from compare.py, field tree_mine) and compare with the object
carried by pickle from the corpus."""
import json, pickle, sys, warnings
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
import discopy
assert discopy.__file__.startswith(sys.argv[1])
from discopy.utils import loads
corpus = json.load(open(sys.argv[3]))
rows = json.load(open(sys.argv[2]))
bad = total = 0
for x, r in zip(corpus, rows):
    if r.get("status") != "ok" or r["tree_mine"].startswith("!"):
        continue
    total += 1
    try:
        obj = pickle.loads(bytes.fromhex(x["pickle"]))
        ok = type(loads(r["tree_mine"])) is type(obj) and loads(r["tree_mine"]) == obj
    except BaseException:
        ok = False
    bad += not ok
print("forward: trees written there that read back here unequal or not at all:", bad, "of", total)
