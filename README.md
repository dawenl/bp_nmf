bp_nmf
======

Beta process NMF (BP-NMF)

**Note**: BP-NMF uses L-BFGS-B from `scipy.optimize <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_ solver to jointly optimize multiple univariate functions, which may lead to numerically-unstable result. For more stable result (but much slower), one can replace L-BFGS-B with a univariate solver on each nonconjugate variable.
