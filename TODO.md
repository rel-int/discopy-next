> Review round on #462 (cubic-dev-ai[bot]), head ed2c7de3:
>
> 1. .claude/skills/next-checks/checks.py:320 (thread 4493258342)
> [P2] `run()` merges stderr into the text that `gate_ruff` parses as JSON, so any ruff warning on stderr turns a clean run into a false FAIL. Capture stdout separately for the ruff call, or parse only the JSON array from the output.
>
> 2. .claude/skills/next-checks/checks.py:53 (thread 4493258361)
> [P2] The sync fallback never runs when `uv` is missing: `subprocess.run` raises `FileNotFoundError` before `proc.returncode` exists, and `main()` does not catch it. Handle the missing executable like a failed sync so the checker emits its normal failure/fallback output instead of a traceback.
>
> ```suggestion
> +            try:
> +                proc = subprocess.run(
> +                    ["uv", "sync", "--locked", "--group", "dev", "--group", "all",
> +                     "--python", "3.14"], cwd=self.root, env=env, stdout=file,
> +                    stderr=subprocess.STDOUT, check=False)
> +            except FileNotFoundError as error:
> +                file.write(f"{error}\n")
> +                proc = subprocess.CompletedProcess(["uv"], 127)
> ```

- [x] .claude/skills/next-checks/checks.py:320 `run()` merges stderr into the text that `gate_ruff` parses as JSON, so any ruff warning on stderr turns a cl…
- [x] .claude/skills/next-checks/checks.py:53 The sync fallback never runs when `uv` is missing: `subprocess.run` raises `FileNotFoundError` before `proc.r…
