forget about making a single unified search. implement the specific search strategy in each of the concrete categories, while keeping it modular. there should be a method for searching generators, another guiding unification of context in different doctrines, etc...
for example, unifying Tensor[A, B] with C depends on the level of the hierarchy. in planar diagrams, it amounts to sampling a split then unifying A with C[:n] and B with C[n:], for symmetric unifying A and B would involve sampling a permutation sigma such that B = sigma(A), same for markov, frobenius, compact, etc... basically plumbing should be actively searched instead of waiting for a structural plumbing generator to give us the correct context.
remove search from pattern.py and simply implement it as a family of possibly abstract methods in abc then implement them in concrete categories. implement focusing only in categories where it makes sense
run an experiment and report whether it helps with the hypothesis statistics, in dead ends or even just overall sampling distribution

go ahead, on a new branch off this one

## Baseline
- [ ] experiment script measuring hypothesis statistics, dead ends and the sampling distribution, run on the unified search

## Search in abc
- [ ] move the search out of `pattern.py` into a family of classmethods of `abc.Category`: `search`, `leaves`, `branches`, `contexts`, `focus`
- [ ] planar contexts: a split, i.e. the unification of the patterns
- [ ] symmetric contexts: a permutation
- [ ] Markov contexts: copies and discards
- [ ] compact contexts: cups and caps
- [ ] hypergraph contexts: spiders
- [ ] focusing only in biclosed categories
- [ ] tests and docs

## Experiment
- [ ] rerun the experiment on the doctrine search and compare
- [ ] full checks, changelog
