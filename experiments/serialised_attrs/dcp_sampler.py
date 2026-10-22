"""pytest plugin: sample live discopy objects during a test run, then write
the repr, tree and pickle of each to $CAPTURE_OUT."""
import gc, json, os, pickle, warnings

SAMPLES, PER_TYPE, EVERY = {}, 40, 10
_count = [0]

def _scan():
    for obj in gc.get_objects():
        t = type(obj)
        mod = t.__dict__.get("__module__", "") if isinstance(t, type) else ""
        mod = mod if isinstance(mod, str) else ""
        if not mod.startswith("discopy") or isinstance(obj, type):
            continue
        bucket = SAMPLES.setdefault(f"{mod}.{t.__qualname__}", [])
        if len(bucket) < PER_TYPE and all(o is not obj for o in bucket):
            bucket.append(obj)

def pytest_runtest_teardown(item, nextitem):
    _count[0] += 1
    if _count[0] % EVERY == 0 or nextitem is None:
        _scan()

def _try(f):
    try:
        return f()
    except BaseException as e:  # noqa
        return "!" + type(e).__name__ + ": " + str(e)[:120]

def pytest_sessionfinish(session, exitstatus):
    from discopy.utils import dumps
    warnings.simplefilter("ignore")
    out, seen = [], set()
    for name, objs in sorted(SAMPLES.items()):
        for obj in objs:
            r = _try(lambda: repr(obj))
            key = (name, r)
            if key in seen:
                continue
            seen.add(key)
            out.append({"type": name, "repr": r,
                        "tree": _try(lambda: dumps(obj)),
                        "pickle": _try(lambda: pickle.dumps(obj).hex())})
    json.dump(out, open(os.environ["CAPTURE_OUT"], "w"))
    print(f"\ncaptured {len(out)} objects of {len(SAMPLES)} types")
