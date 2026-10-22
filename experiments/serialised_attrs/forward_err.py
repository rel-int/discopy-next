import json, pickle, sys, warnings, collections, re
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
import discopy; assert discopy.__file__.startswith(sys.argv[1])
from discopy.utils import loads
corpus=json.load(open(sys.argv[3])); rows=json.load(open(sys.argv[2])); c=collections.Counter(); t=collections.Counter()
for x, r in zip(corpus, rows):
    if r.get("status")!="ok" or r["tree_mine"].startswith("!"): continue
    try:
        obj=pickle.loads(bytes.fromhex(x["pickle"])); y=loads(r["tree_mine"])
        if type(y) is type(obj) and y==obj: continue
        c["(loads, unequal)"]+=1
    except BaseException as e:
        c[re.sub(r"'[^']*'","'…'",type(e).__name__+": "+str(e))[:100]]+=1
    t[r["type"]]+=1
for k,v in c.most_common(8): print(f"{v:5d}  {k}")
for k,v in t.most_common(8): print(f"{v:5d}  {k}")
