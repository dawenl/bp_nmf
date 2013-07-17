Beta Process Sparse NMF (BP-NMF)
======

A Bayesian nonparametric extension of Nonnegative Matrix Factorization (NMF). 

**Note**: BP-NMF uses L-BFGS-B solver from [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b) to jointly optimize multiple univariate functions, which may lead to numerically-unstable result. For more stable result (but much slower), one can replace L-BFGS-B with a univariate solver on each nonconjugate variable.

### Folder structure:
* code: contains the code for inference, utils, the experiments. **Note**: All the files with the name `exp_*.py` are meant to run with [IPython Notebook](http://ipython.org/notebook.html). Also, [librosa](https://github.com/bmcfee/librosa) is required for all the signal processing components in the experiments sciprts. 
* notes: a detailed derivation of the full variational inference

### Dependencies:
* numpy 
* scipy
* [librosa](https://github.com/bmcfee/librosa) (for signal processing components in the experiments)
* [python-midi](https://github.com/vishnubob/python-midi) (for blind source separation experiments)
