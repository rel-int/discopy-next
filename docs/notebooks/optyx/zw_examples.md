---
title: Zw Examples
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# Evaluating photonic circuits using the permanent method and tensor networks
<!---->
By representing the circuits in $\bf{ZW}_\infty$, we can treat them as tensor networks which can be contracted using different methods: the permanent of the unitary, Perceval's simulator, or a tensor network contracted by DisCoPy or Quimb.

```python {.marimo}
import numpy as np

from discopy.optyx import photonic
from discopy.optyx.core.backends import (
    DiscopyBackend, PercevalBackend, PermanentBackend, QuimbBackend)
```

## Hong-Ou-Mandel effect
<!---->
We can define the optical circuit for the Hong-Ou-Mandel effect as a photonic diagram:

```python {.marimo}
hom = photonic.Create(1, 1) >> photonic.BS >> photonic.Select(1, 1)
```

```python {.marimo}
hom.draw()
```

Its Kraus map is a $\bf{ZW}_\infty$ diagram:

```python {.marimo}
hom.get_kraus().draw()
```

Evaluating the permanent, and with Perceval, which gives the probability rather than the amplitude:

```python {.marimo}
hom.eval(PermanentBackend()).tensor.array
```

```python {.marimo}
hom.eval(PercevalBackend()).tensor.array
```

Evaluating the tensor network:

```python {.marimo}
hom.eval(DiscopyBackend()).tensor.array
```

# Number operator

```python {.marimo}
number_operator = photonic.NumOp()
number_operator.get_kraus().draw()
```

On $n$ photons, the number operator gives the amplitude $n$:

```python {.marimo}
def number(n):
    return photonic.Create(n) >> number_operator >> photonic.Select(n)

for _n in (2, 3, 12):
    assert np.isclose(number(_n).eval(PermanentBackend()).tensor.array, _n)
    assert np.isclose(number(_n).eval(DiscopyBackend()).tensor.array, _n)

number(12).eval(DiscopyBackend()).tensor.array
```

## More complicated circuit

```python {.marimo}
circ = (
    photonic.Create(1, 1, 1, 1)
    >> photonic.BBS(0.3) @ photonic.MZI(0.34, 0.10)
    >> photonic.Phase(1 / 4) @ photonic.Id(2) @ photonic.Phase(1 / 8)
)
```

```python {.marimo}
circ.draw()
```

```python {.marimo}
circ.get_kraus().draw()
```

The four methods give the same distribution over the output occupations:

```python {.marimo}
distributions = [
    circ.eval(backend).prob_dist()
    for backend in (
        PercevalBackend(), PermanentBackend(), DiscopyBackend(), QuimbBackend())]
occupations = set().union(*distributions)

assert all(
    np.isclose(distribution.get(key, 0), distributions[0].get(key, 0))
    for distribution in distributions for key in occupations)

dict(sorted(distributions[0].items()))
```
