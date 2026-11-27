`pytest --help` no longer says `--skip-extra` skips nothing once the extras
are installed: the flag skips its fixed list whether or not they import
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
