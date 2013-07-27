"""
Python translation of MATALB code for GaP-NMF as in:

    Bayesian Nonparametric Matrix Factorization for Recorded Music

    by Matthew D. Hoffman et al. in ICML 2010

The original MATLAB code:
    http://www.cs.princeton.edu/~mdhoffma/code/gapnmfmatlab.tar

CREATED: 2013-07-23 14:16:41 by Dawen Liang <daliang@adobe.com>

"""

import functools
import time

import numpy as np
from matplotlib import pyplot as plt

import _gap

specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class GaP_NMF:
    def __init__(self, X, K=100, smoothness=100, seed=None, **kwargs):
        """ gap = GaP_NMF(X, K=100, smoothness=100, seed=None, a=0.1, b=0.1,
                          alpha=1.)

        Required arguments:
            X:              F-by-T nonnegative matrix (numpy.ndarray)
                            the data to be factorized

        Optional arguments:
            K:              the size of the initial components

            smoothness:     control the concentration of the variational
                            parameters

            seed:           random seed if None

            a, b, alpha:    hyperparameters

        """

        self.X = X / np.mean(X)
        self.K = K
        self.F, self.T = X.shape
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.a = kwargs['a'] if 'a' in kwargs else 0.1
        self.b = kwargs['b'] if 'b' in kwargs else 0.1
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.

    def _init(self, smoothness):
        self.rhow = 10000 * np.random.gamma(smoothness, 1./smoothness,
                                            size=(self.F, self.K))
        self.tauw = 10000 * np.random.gamma(smoothness, 1./smoothness,
                                            size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness, 1./smoothness,
                                            size=(self.K, self.T))
        self.tauh = 10000 * np.random.gamma(smoothness, 1./smoothness,
                                            size=(self.K, self.T))
        self.rhot = self.K * 10000 * np.random.gamma(smoothness, 1./smoothness,
                                                     size=(self.K, ))
        self.taut = 1./self.K * 10000 * np.random.gamma(smoothness,
                                                        1./smoothness,
                                                        size=(self.K, ))
        self.compute_expectations()

    def compute_expectations(self):
        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.a, self.rhow,
                                                            self.tauw)
        self.Ewinvinv = 1./self.Ewinv
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b, self.rhoh,
                                                            self.tauh)
        self.Ehinvinv = 1./self.Ehinv
        self.Et, self.Etinv = _gap.compute_gig_expectations(self.alpha/self.K,
                                                            self.rhot,
                                                            self.taut)
        self.Etinvinv = 1./self.Etinv

    def update(self):
        ''' Do optimization for one iteration
        '''
        self.update_h()
        self.update_w()
        self.update_theta()
        # truncate unused components
        self.clear_badk()

    def update_w(self):
        goodk = self.goodk()
        xxtwidinvsq = self.X * self._xtwid(goodk)**(-2)
        xbarinv = self._xbar(goodk) ** (-1)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]
        self.rhow[:, goodk] = self.a + np.dot(xbarinv, dEt *
                                              self.Eh[goodk, :].T)
        self.tauw[:, goodk] = self.Ewinvinv[:, goodk]**2 * \
                np.dot(xxtwidinvsq, dEtinvinv * self.Ehinvinv[goodk, :].T)
        self.tauw[self.tauw < 1e-100] = 0
        self.Ew[:, goodk], self.Ewinv[:, goodk] = _gap.compute_gig_expectations(
            self.a, self.rhow[:, goodk], self.tauw[:, goodk])
        self.Ewinvinv[:, goodk] = 1./self.Ewinv[:, goodk]

    def update_h(self):
        goodk = self.goodk()
        xxtwidinvsq = self.X * self._xtwid(goodk)**(-2)
        xbarinv = self._xbar(goodk) ** (-1)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]
        self.rhoh[goodk, :] = self.b + np.dot(dEt[:, np.newaxis] *
                                              self.Ew[:, goodk].T, xbarinv)
        self.tauh[goodk, :] = self.Ehinvinv[goodk, :]**2 * \
                np.dot(dEtinvinv[:, np.newaxis] * self.Ewinvinv[:, goodk].T,
                       xxtwidinvsq)
        self.tauh[self.tauh < 1e-100] = 0
        self.Eh[goodk, :], self.Ehinv[goodk, :] = _gap.compute_gig_expectations(
            self.b, self.rhoh[goodk, :], self.tauh[goodk, :])
        self.Ehinvinv[goodk, :] = 1./self.Ehinv[goodk, :]

    def update_theta(self):
        goodk = self.goodk()
        xxtwidinvsq = self.X * self._xtwid(goodk)**(-2)
        xbarinv = self._xbar(goodk) ** (-1)
        self.rhot[goodk] = self.alpha + np.sum(np.dot(self.Ew[:, goodk].T,
                                                      xbarinv) *
                                               self.Eh[goodk, :], axis=1)
        self.taut[goodk] = self.Etinvinv[goodk]**2 * \
                np.sum(np.dot(self.Ewinvinv[:, goodk].T, xxtwidinvsq) *
                       self.Ehinvinv[goodk, :], axis=1)
        self.taut[self.taut < 1e-100] = 0
        self.Et[goodk], self.Etinv[goodk] = _gap.compute_gig_expectations(
            self.alpha/self.K, self.rhot[goodk], self.taut[goodk])
        self.Etinvinv[goodk] = 1./self.Etinv[goodk]

    def goodk(self, cut_off=None):
        if cut_off is None:
            cut_off = 1e-10 * np.amax(self.X)

        powers = self.Et * np.amax(self.Ew, axis=0) * np.amax(self.Eh, axis=1)
        sorted = np.flipud(np.argsort(powers))
        idx = np.where(powers[sorted] > cut_off * np.amax(powers))[0]
        goodk = sorted[:(idx[-1] + 1)]
        if powers[goodk[-1]] < cut_off:
            goodk = np.delete(goodk, -1)
        return goodk

    def clear_badk(self):
        ''' Set unsued components' posteriors equal to their priors
        '''
        goodk = self.goodk()
        badk = np.setdiff1d(np.arange(self.K), goodk)
        self.rhow[:, badk] = self.a
        self.tauw[:, badk] = 0
        self.rhoh[badk, :] = self.b
        self.tauh[badk, :] = 0
        self.compute_expectations()

    def figures(self):
        ''' Animation-type of figures can only be created with PyGTK backend
        '''
        plt.subplot(3, 2, 1)
        specshow(np.log(self.Ew))
        plt.title('E[W]')
        plt.xlabel('component index')
        plt.ylabel('frequency')

        plt.subplot(3, 2, 2)
        specshow(np.log(self.Eh))
        plt.title('E[H]')
        plt.xlabel('time')
        plt.ylabel('component index')

        plt.subplot(3, 2, 3)
        plt.bar(np.arange(self.K), self.Et)
        plt.title('E[theta]')
        plt.xlabel('component index')
        plt.ylabel('E[theta]')

        plt.subplot(3, 2, 5)
        specshow(np.log(self.X))
        plt.title('Original Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        plt.subplot(3, 2, 6)
        specshow(np.log(self._xbar()))
        plt.title('Reconstructed Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        time.sleep(0.000001)

    def bound(self):
        score = 0
        goodk = self.goodk()

        xbar = self._xbar(goodk)
        xtwid = self._xtwid(goodk)
        score -= np.sum(self.X / xtwid + np.log(xbar))
        score += _gap.gig_gamma_term(self.Ew, self.Ewinv, self.rhow, self.tauw,
                                     self.a, self.a)
        score += _gap.gig_gamma_term(self.Eh, self.Ehinv, self.rhoh, self.tauh,
                                     self.b, self.b)
        score += _gap.gig_gamma_term(self.Et, self.Etinv, self.rhot, self.taut,
                                     self.alpha/self.K, self.alpha)
        return score

    def _xbar(self, goodk=None):
        if goodk is None:
            goodk = np.arange(self.K)
        dEt = self.Et[goodk]
        return np.dot(self.Ew[:, goodk], dEt[:, np.newaxis] *
                      self.Eh[goodk, :])

    def _xtwid(self, goodk):
        dEtinvinv = self.Etinvinv[goodk]
        return np.dot(self.Ewinvinv[:, goodk], dEtinvinv[:, np.newaxis] *
                      self.Ehinvinv[goodk, :])




