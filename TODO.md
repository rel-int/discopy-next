> Review round on #462 (cubic-dev-ai[bot]), head 30ece97d:
>
> 1. .claude/skills/next-checks/checks.py:138 (thread 4493328796)
> [P2] `venv()` calls the fallback `.venv/bin/python` without handling a missing file. If `uv` is absent and no `.venv` exists, this raises an uncaught `FileNotFoundError` before any gate prints. Wrap this call in the same `FileNotFoundError` handling used for `uv sync`, or fall back to a version of "unknown".
>
> 2. .claude/skills/next-checks/checks.py:199 (thread 4493328803)
> [P3] Write cache entries atomically; a partial write can make every subsequent run fail with an uncaught `JSONDecodeError` instead of treating the entry as a cache miss. Write to a uniquely named temporary file and replace `path` only after serialization completes.
>
> ```suggestion
>         fd, temporary = tempfile.mkstemp(dir=directory)
>         try:
>             with os.fdopen(fd, "w", encoding="utf-8") as file:
>                 json.dump(value, file)
>             pathlib.Path(temporary).replace(path)
>         finally:
>             pathlib.Path(temporary).unlink(missing_ok=True)
> ```

- [x] .claude/skills/next-checks/checks.py:138 `venv()` calls the fallback `.venv/bin/python` without handling a missing file.
- [x] .claude/skills/next-checks/checks.py:199 Write cache entries atomically; a partial write can make every subsequent run fail with an uncaught `JSONDeco…
