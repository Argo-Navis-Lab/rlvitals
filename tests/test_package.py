import subprocess
import sys
from importlib.metadata import requires, version

import rlvitals


def test_the_version_is_the_installed_one() -> None:
    assert rlvitals.__version__ == version("rlvitals") == "0.1.0"


def test_the_package_depends_on_nothing() -> None:
    assert requires("rlvitals") is None


def test_importing_it_loads_no_heavy_package() -> None:
    script = (
        "import sys, rlvitals; print(sorted({'torch', 'numpy', 'transformers'} & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
