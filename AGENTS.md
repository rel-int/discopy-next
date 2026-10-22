# AGENTS.md

## What

DisCoPy is a Python toolkit for computing with string diagrams.

Read the following documents (<10k tokens) before any work on the package:

- @RULES.md describes the collaboration and Git protocol for shared PR branches, follow it exactly.
- @STYLE.md describes coding guidelines that all your work should try to follow.
- @README.md contains a high-level description of the features along with some examples.
- @CONTRIBUTING.md contains setup instructions and our general coding philosophy.

## Next

This repo is the *next* repo of [discopy/discopy](https://github.com/discopy/discopy), which stays fully human-reviewed: here agents merge into `main` themselves, and their job is to keep the drift between the two small. The rules are the `Next repos` section of [toumix/desire](https://github.com/toumix/desire/blob/main/AGENTS.md#next-repos); in short:

- `main` is upstream `main` plus work meant to go upstream: agents merge into it once its checks pass and its `TODO.md` is deleted, with no human sign-off: the human approval of `CONTRIBUTING.md`'s LLM guidelines binds upstream, not here
- down: upstream `main` is merged into `main` every night, never rebased
- up: an upstream pull request is opened only when a human asks, cut from upstream `main` and small enough to read; agents propose the slices on [#55](https://github.com/rel-int/discopy-next/issues/55)
- drift is `git diff upstream/main...main -- . ':!.github'`, split `discopy/`, `test/` and `docs/`, and tended down by slicing: moving an open upstream pull request here raises the measured drift but adds nothing to read, it only moves where the work waits
- experiments do not live on long branches: a big, speculative one gets a repository of its own, a small one lands in `main` as a notebook
- `#N` is this repo, `discopy/discopy#N` upstream

Each human's agents move their own open work from upstream, never somebody else's; [#2](https://github.com/rel-int/discopy-next/issues/2) is the record of the first such move:

1) triage every open agent pull request upstream in one issue here: **next** (merge or port), **own repo** or **close**, with its size and how far behind it is, and let its human rule on it
2) merge the ones that merge as they stand, a few per pull request, each as a merge commit of the upstream head so its commits survive, each conflict resolved once; `CHANGELOG.md` by hand, since git's union driver brings back lines one side deleted
3) port the ones that do not merge in a pull request each, carrying the upstream description
4) close the upstream pull request of each **next** row with a comment pointing to the pull request here that merged it, and an **own repo** or **close** row only once its human agrees, with a comment saying where it went or why it closed
5) re-file each open agent issue here, its body headed `Ported from discopy/discopy#N (<author>, <date>)`, and close the upstream one with `Moved to rel-int/discopy-next#M`

## Where

- [discopy](discopy/) contains the code with a lot of modules and a few submodules
- [discopy.abc](discopy/abc.py) contains abstract base classes for each level of the hierarchy
- [discopy.axioms](discopy/axioms.py) contains the property-testing module: `Axiom` states a law once on an abstract base class, `Testable` generates the terms it quantifies over, and the matrix in `proptest/` checks every cell. Read it before adding a category or an axiom.
- [test](test/) is flat with one file for each module, mirroring `discopy/`; submodules (`grammar`, `quantum`, `drawing`, `python`) keep their own directory with one file per submodule, and shared test data lives in `fixtures`
- [docs](docs/) contains notebooks and pictures generated automatically when running the tests

## How

Before writing any code, make sure that:

1) your change was first described in high-level mathematical terms
2) this description aligns with the data structures you plan to use

Before pushing anything, make sure that:

- you have reported any bugs or confusing docs that you encounter even if unrelated
- you have added docs and tests that are complete but concise as best as you can
- you have added an entry to the `[Unreleased]` section of @CHANGELOG.md for any user-facing change
- you have `uv run` `pflake8 discopy`, `pylint discopy` and `coverage run -m pytest` as described in @CONTRIBUTING.md
- you have respected the [code style guide](CONTRIBUTING.md#code-style-guide)

A pull request is **ready for sign-off** when all three of these hold:

1) its `TODO.md` is deleted, every point having been `[x]` or filed as an issue
2) CI is green on the real jobs, with the target branch merged in
3) no review thread is waiting on an agent, i.e. every thread is either resolved or waiting on human feedback
