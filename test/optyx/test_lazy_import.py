import subprocess
import sys


def test_lazy_extras_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy, discopy.optyx; "
        "imported = {name.split('.')[0] for name in sys.modules}; "
        "assert not imported & {'perceval', 'graphix', 'pytket', 'qiskit'}"
    ], check=True)
