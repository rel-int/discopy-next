cubic on `36a76583`, `test/optyx/test_backends.py:939`:

> P2: This regression test raises `TypeError` when comparing the complex amplitudes, so the permanent-backend tests cannot pass even when the backend is correct. Compare the result arrays with `np.allclose`, or update the helper to compare complex values by magnitude.

It does not raise: `math.isclose` takes a `numpy.complex128` through `__float__`, warning `ComplexWarning` and dropping the imaginary part. So `dict_allclose` compares real parts only, and an amplitude of `0.707j` matches any other imaginary number.

- [WIP] @session_01MEJxy43YMF4nTF8NEcP8Je-2026-11-06 02:47 `dict_allclose` compares complex amplitudes, in `test_backends.py` and its copy in `test_perceval.py`
