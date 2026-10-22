"""The figure scripts of the notes: all compile, and the cheap ones run (in a temporary copy, so that the
figures are written to <tmp>/figures/ and not into the repository)."""
import glob
import os
import py_compile
import shutil
import subprocess
import sys

import pytest

from conftest import COURSE_CODE

SCRIPTS = sorted(glob.glob(os.path.join(COURSE_CODE, "*.py")))
CHEAP = {"appA_figures.py": ["appA/distributions.pdf"],
         "ch04_landscape.py": ["ch04/landscape.tikz"],
         "ch05_landscape.py": ["ch05/landscape.tikz"]}


@pytest.mark.parametrize("path", SCRIPTS, ids=os.path.basename)
def test_compiles(path, tmp_path):
    py_compile.compile(path, cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.fixture(scope="module")
def course_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("notes")
    shutil.copytree(COURSE_CODE, root / "course_code", ignore=shutil.ignore_patterns("__pycache__"))
    for sub in ("appA", "ch04", "ch05"):
        (root / "figures" / sub).mkdir(parents=True)
    return root


@pytest.mark.parametrize("script", sorted(CHEAP))
def test_cheap_script_runs(script, course_copy):
    if not os.path.exists(os.path.join(COURSE_CODE, script)):
        pytest.skip(f"{script} not in course_code")
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, script], cwd=course_copy / "course_code", env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    for out in CHEAP[script]:
        assert (course_copy / "figures" / out).stat().st_size > 0
