"""Shared setup of the smoke tests: the course code is a folder of plain modules, put on the search path here."""
import os
import sys

# The test problems are tiny: multithreaded BLAS only adds overhead (and is very slow on oversubscribed machines).
# Set before NumPy is imported; inherited by the subprocesses of the tests.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSE_CODE = os.path.join(ROOT, "course_code")
PROJECTS = os.path.join(ROOT, "projects")
sys.path.insert(0, COURSE_CODE)


@pytest.fixture(scope="session")
def deconv():
    """The deconvolution example of the notes (n = 128, m = 64) with noisy data of both test signals."""
    import running_examples as re_
    A, t, s = re_.deconvolution_operator()
    rng = np.random.default_rng(0)
    y = {name: re_.noisy_data(A, f(t), re_.NOISE_STD, rng) for name, f in re_.SIGNALS.items()}
    return A, t, s, y
