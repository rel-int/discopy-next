# E-graph traces

The history of 50 e-graphs, without values: test data for an e-graph builder
([#499](https://github.com/rel-int/discopy-next/issues/499)). Each comes from one task of
[1D-ARC](https://github.com/khalil-research/1D-ARC) (at `1e74dc4`), searched for a program with no
model and a budget of 51,200 evaluations. The search grows one e-graph per task: it evaluates a
primitive on classes, and two programs with equal values on every input of the task are one class.

- **Tasks:** the 25 of the 901 with the most e-nodes, and 25 of the others drawn at random
  (`index.json` says which). Files are numbered by size, `trace-00` the smallest.
- **Sizes:** 52 to 19,764 e-nodes, 41 to 4,566 classes; 1.7 MB gzipped in all.
- **Nothing of the search is here:** heads are renamed `h0, h1, ..` in order of first use and only
  their arity is kept, and there are no values, tasks or task names.

## Format

`trace-NN.json.gz` is one gzipped JSON object:

| key | what |
| --- | --- |
| `heads` | `[{"name": "h0", "arity": 0}, ...]`, indexed by the head numbers of `rows` |
| `rows` | the e-nodes in the order the search added them, each `[head, out, *args]` |
| `classes` | the number of classes at the end |
| `enodes` | `len(rows)` |
| `evaluations` | the evaluations the search spent, failed ones included |
| `egraph_seconds`, `search_seconds` | see [Timings](#timings) |

A row `[head, out, *args]` says: `add(head, args)` gave class `out`. Classes are numbered in order
of creation, from 0. So:

- a row whose `out` is the next unused number **created** a class;
- any other row **merged**: its value was that of the existing class `out`. For a builder this is
  `merge(add(head, args), out)`.

The first rows have no argument: the base classes (the input and the constants), each with its own
head. An argument is always a class created by an earlier row.

**The final partition** is the column `out`: row `i` is an e-node of class `rows[i][1]`, and
`classes` counts the distinct values. No pair `(head, args)` occurs twice, and a class never
merges with another once both exist, so replaying a trace never has to repair a parent: it
exercises `add`, the hash-cons and `merge` of a fresh class, not the cascade of a rebuild.

## Timings

- `egraph_seconds`: the time the search's own bookkeeping takes on the trace, replayed without
  values, best of 7. That bookkeeping is a `dict` from `(head, args)` to the class and two lists,
  `enodes` and `kinds`; `out` is only what the replay returns:

  ```python
  def replay(trace):
      cache, enodes, kinds, out = {}, [], [], []
      for row in trace["rows"]:
          key = (row[0], tuple(row[2:]))
          cid = cache.get(key)
          if cid is None:
              cid = row[1]
              if cid == len(kinds):
                  kinds.append(row[0])
              cache[key] = cid
              enodes.append(key + (cid,))
          out.append(cid)
      return out
  ```

  The search also hashes each value vector to find its class; a trace has no values, so that part
  is not in the number.
- `search_seconds`: the whole search, running the programs included. On the largest trace the
  bookkeeping is 0.009 s of 5.0 s.

Both were measured on one 4-core cloud sandbox, CPython 3.12. They are not comparable across
machines: a benchmark should time `replay` again where it runs.
