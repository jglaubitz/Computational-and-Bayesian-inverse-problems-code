# Computational and Bayesian Inverse Problems: course code

[![tests](https://github.com/jglaubitz/Computational-and-Bayesian-inverse-problems-code/actions/workflows/tests.yml/badge.svg)](https://github.com/jglaubitz/Computational-and-Bayesian-inverse-problems-code/actions/workflows/tests.yml)

Python code of the PhD course *Computational and Bayesian Inverse Problems*, Linköping University, 26–30 October 2026 (Jan Glaubitz).

* **Lecture notes:** distributed to the course participants

The repository contains

* `course_code/`: the modules of the lecture notes (forward models of the two running examples, regularization solvers, priors, MCMC samplers, approximate inference) and the scripts that reproduce every computed figure of the notes;
* `projects/`: the starter code and synthetic data of the three default projects;
* `tests/`: fast smoke tests (under a minute), run on every push by GitHub Actions.

The code is written to be read: it favors clarity over speed, and every function has a docstring.


## Installation

Python 3.10 or newer. In a fresh virtual environment:

```bash
git clone https://github.com/jglaubitz/Computational-and-Bayesian-inverse-problems-code.git
cd Computational-and-Bayesian-inverse-problems-code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, scipy, matplotlib, jax (CPU), scikit-image, pytest
pip install blackjax               # optional: only for the NUTS cross-check in ch04_figures.py
pytest                             # smoke tests, under a minute
```

`pip install jax` installs the CPU version of JAX; no GPU is needed.
The code was tested on Linux with Python 3.11 (NumPy 2.4, SciPy 1.17, Matplotlib 3.11, JAX 0.10, scikit-image 0.26) and Python 3.13 (NumPy 2.5, SciPy 1.18, Matplotlib 3.11, JAX 0.12, scikit-image 0.26, BlackJAX 1.7).


## Structure

```
course_code/
    running_examples.py    deconvolution operator and test signals; steady shallow-water model (NumPy) and test bottoms
    swe_jax.py             bathymetry forward model in JAX (differentiable, compiled): make_forward(x_cells, x_obs)
    solvers.py             difference matrices, Tikhonov, CGLS, l1/TV by FISTA, Gauss-Newton, linear-Gaussian posterior, IAS
    priors.py              Gaussian, Laplace, and hierarchical (inverse-gamma) priors: precision matrices and samplers
    samplers.py            random-walk Metropolis, perturb-then-optimize, Gibbs samplers, HMC (NumPy and JAX),
                           elliptical slice sampling; autocorrelation, IAT, ESS, R-hat
    approximations.py      low-rank Laplace approximation, Gaussian VI, Pareto k-hat and importance sampling,
                           variational Bayes for the hierarchical prior, ensemble Kalman update, ES-MDA, EKI
    chXX_figures.py        figures of Chapter XX;  appA_figures.py: figure of Appendix A
    ch04_landscape.py, ch05_landscape.py     TikZ cartoons opening Chapters 4 and 5
figures/                   output folder of the figure scripts (contents are git-ignored)
projects/
    project_A/             Variational Bayes for sparsity-promoting hierarchical models
    project_B/             Bayesian atom-conserving chemical reaction neural networks
    project_C/             Bayesian bathymetry inference from shallow-water data
tests/                     pytest smoke tests
```

To use the modules in your own scripts, start Python in `course_code/` or add it to the search path:

```python
import sys; sys.path.insert(0, "path/to/course_code")
from running_examples import deconvolution_operator
from samplers import gibbs_hierarchical
help(gibbs_hierarchical)
```


## Reproducing the figures of the notes

Every computed figure of the notes is produced by the script of its chapter.
Run the scripts from anywhere; they write to `figures/<chapter>/` at the top level of the repository:

```bash
cd course_code
python ch01_figures.py                    # all figures of Chapter 1 -> figures/ch01/*.pdf
python ch04_figures.py mhidea hmcidea     # selected figures of Chapter 4 only
```

| Chapter | Script | Output | Needs JAX | Run time |
|---|---|---|---|---|
| 1 Inverse problems: the basics | `ch01_figures.py` | `figures/ch01/` | no (scikit-image for `ct.pdf`) | 15 s |
| 2 Deterministic regularization | `ch02_figures.py` | `figures/ch02/` | yes | 30 s |
| 3 Bayesian inverse problems and MAP estimation | `ch03_figures.py` | `figures/ch03/` | yes | 40 s |
| 4 Uncertainty quantification by sampling | `ch04_figures.py`, `ch04_landscape.py` | `figures/ch04/` | yes (BlackJAX optional) | 5–12 min |
| 5 Beyond MCMC: approximate Bayesian inference | `ch05_figures.py`, `ch05_landscape.py` | `figures/ch05/` | yes | 2–3 min |
| A Probability refresher | `appA_figures.py` | `figures/appA/` | no | 10 s |

Run times were measured on a shared two-core cloud machine (Python 3.13, single-threaded BLAS); a laptop is usually faster.
The matrices of the course are small, so a multithreaded BLAS mostly adds overhead; on machines with few cores or under load, `export OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`) before running a script can make it several times faster.
`ch04_figures.py` and `ch05_figures.py` take the names of single figures as arguments (listed in the docstring of each script); without arguments they make all figures.
`ch05_figures.py vibathy` prints the numbers of Gaussian variational inference on the bathymetry problem quoted in the notes instead of drawing a figure.
All random numbers have fixed seeds; other versions of NumPy, SciPy, or JAX may change the last digits of printed numbers, and the installed fonts change the tick labels and line breaks of the figures.
Drawings that are not computed (TikZ sketches, photographs) are part of the notes, not of this repository.


## The default projects

Each folder `projects/project_X/` contains the starter code, a `README.md` that documents every file and data set, a `requirements.txt`, and the synthetic data in `data/`.
The project descriptions are distributed to the course participants.
The starter code contains forward models, priors, and data, but no inference code: that is the project.

| Project | Forward model | Data |
|---|---|---|
| A: variational Bayes for sparsity-promoting hierarchical models | `hyperpriors.py` (inverse gamma, gamma, generalized gamma, half-Cauchy hyper-priors), `model.py` (log densities, prior normalization), `burgers.py`, `burgers_jax.py` (Godunov solver for the inviscid Burgers equation) | deconvolution (linear), Burgers (nonlinear) |
| B: Bayesian atom-conserving chemical reaction neural networks | `acrnn.py` (atom-conservation basis, mass-action CRNN, ODE solves with SciPy) | trimerization: standard, scarce, and noisy data sets |
| C: Bayesian bathymetry inference from shallow-water data | `swe.py` (steady states, well-balanced finite volumes), `flume.py` (wave flume, NumPy and JAX), `priors.py` | steady channel (dense, sparse sensors); flume (hill, box) |

Work inside the project folder, e.g.,

```bash
cd projects/project_C
python -c "import numpy as np, swe; d = np.load('data/steady_smooth.npz'); print(sorted(d.files))"
```

and see the minimal examples in the project's `README.md`.
Each project is a folder of plain modules; project C has its own `priors.py`, which is different from `course_code/priors.py`, so do not put both folders on the search path at the same time.

`generate_data.py` regenerates the data of a project; with the default seed it reproduces the shipped files (checked by the tests).
It **overwrites** the files in `data/` and writes an overview figure to `projects/figures/`; to keep the shipped data, run it in a copy of the folder.
The data files also contain the truth (marked "for verification only" in the READMEs): use it to assess your results, never in the inference.
The measured flume data of Angel et al. used in project C are not included; they are available at <https://doi.org/10.15480/882.9403> under CC BY-NC 4.0.


## Tests

```bash
pytest            # under a minute
```

The tests import every module, run tiny versions of the key computations (deconvolution operator and its condition number, Tikhonov, CGLS, FISTA, IAS, Gibbs, HMC in NumPy and JAX, elliptical slice sampling, diagnostics, low-rank Laplace, Gaussian VI, CAVI, ensemble Kalman methods, the bathymetry forward model in NumPy and JAX), check that all figure scripts compile and that the cheap ones run, and run the project starter code and data generators.
GitHub Actions runs them on Ubuntu with Python 3.11 and 3.13 (`.github/workflows/tests.yml`).


## License

The code is released under the MIT License (see `LICENSE`).
The synthetic data in `projects/*/data/` are released under the same license.
The lecture notes are not part of this repository and are licensed separately.


## How to cite

If you use this code, please cite the lecture notes:

```bibtex
@misc{Glaubitz2026CBIP,
  author       = {Glaubitz, Jan},
  title        = {Computational and {B}ayesian Inverse Problems},
  howpublished = {Lecture notes, PhD course, Link{\"o}ping University},
  year         = {2026},
  note         = {Code: \url{https://github.com/jglaubitz/Computational-and-Bayesian-inverse-problems-code}}
}
```


## Use of generative AI

Generative AI (Claude, Anthropic) was used to help prepare the lecture notes and the code in this repository, including this README.
The author is responsible for the content.
