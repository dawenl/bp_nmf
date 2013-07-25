Beta Process Sparse NMF (BP-NMF)
======

A Bayesian nonparametric extension of Nonnegative Matrix Factorization (NMF). 

**Note**: BP-NMF uses L-BFGS-B solver from [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b) to jointly optimize multiple univariate functions, which may lead to numerically-unstable result. For more stable result (but much slower), one can replace L-BFGS-B with a univariate solver on each nonconjugate variable.

**Note of Note**: Since for all the nonconjugate variables update, we are essentially solving an numerical optimization problem with L-BFGS (and haven't figured out a way to do multiplicative-type of update yet), BP-NMF can take quite a while if the input matrix is large (> 2 minutes of 22.05 kHz signals with 1024-point DFT and 50% overlap). Try not to process a huge recording.

### What's included:
#### code/ 
Contains the code for inference, utils, the experiments. **Note**: All the files with the name `exp_*.py` are meant to run with [IPython Notebook](http://ipython.org/notebook.html). Also, [librosa](https://github.com/bmcfee/librosa) is required for all the signal processing components in the experiments sciprts. 

There is also a Python translation of the gamma process NMF (GaP-NMF) where the original MATLAB code is developed by [Matt Hoffman](http://www.cs.princeton.edu/~mdhoffma/).
#### notes/ 
A detailed derivation of the full variational inference.

### Dependencies:
* numpy 
* scipy
* [librosa](https://github.com/bmcfee/librosa) (for signal processing components in the experiments)
* [python-midi](https://github.com/vishnubob/python-midi) (for blind source separation experiments)
