The `test` extra includes `notebooks`, so `pip install -e '.[test]'` brings
anywidget and the "Build without uv" recipe's `coverage run -m pytest` no
longer stops at collection on `discopy/drawing/widget.py` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
