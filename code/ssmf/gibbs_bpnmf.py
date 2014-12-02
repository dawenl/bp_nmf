#!/usr/bin/env python
"""
Beta-process NMF with Gibbs sampler

CREATED: 2014-11-25 16:37:39 by Dawen Liang <dliang@ee.columbia.edu>

"""

import numpy as np
import scipy.stats

import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)


class Gibbs_BP_NMF(BaseEstimator, TransformerMixin):
    '''
    Stochastic structured mean-field variational inference for Beta process
    Poisson NMF
    '''
    def __init__(self, n_components=500, n_samples=100, burn_in=1000,
                 n_lags=10, cutoff=1e-3, smoothness=100, random_state=None,
                 verbose=False, **kwargs):
        self.n_components = n_components
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.n_lags = n_lags
        self.cutoff = cutoff
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        # hyperparameters for components
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

        # hyperparameters for activation
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

        # hyperparameters for sparsity on truncated beta process
        self.a0 = float(kwargs.get('a0', 1.))
        self.b0 = float(kwargs.get('b0', 1.))

    def fit(self, X):
        n_feats, n_samples = X.shape
        # randomly initialize parameters
        self.W = np.random.gamma(self.a, 1. / self.b,
                                 size=(n_feats, self.n_components))
        self.H = np.random.gamma(self.c, 1. / self.d,
                                 size=(self.n_components, n_samples))
        self.pi = np.random.beta(self.a0, self.b0, size=self.n_components)

        # randomly initalize binary mask
        self.S = (np.random.rand(self.n_components, n_samples) > .5)
        self._gibbs_sample_burnin(X)
        return self

    def _gibbs_sample_burnin(self, X):
        self.log_ll = np.zeros(self.burn_in)
        if self.verbose:
            print 'Gibbs burn-in'
            sys.stdout.flush()
        for b in xrange(self.burn_in):
            self._sample(X)
            self.log_ll[b] = self._log_likelihood(X)
            if self.verbose:
                sys.stdout.write('\r\tIteration: %d\tLog_ll: %.3f' %
                                 (b, self.log_ll[b]))
                sys.stdout.flush()

    def _sample(self, X):
        self._gibbs_sample_S(X)
        self._gibbs_sample_WH(X)
        pass

    def _gibbs_sample_S(self, X):
        for k in xrange(self.n_components):
            X_neg_k = self.W.dot(self.H * self.S) - \
                np.outer(self.W[:, k], self.H[k] * self.S[k])
            log_Ph = np.log(self.pi[k]) + \
                np.sum(X * np.log(X_neg_k + np.outer(self.W[:, k], self.H[k]))
                       - np.outer(self.W[:, k], self.H[k]), axis=0)
            log_Pt = np.log(1 - self.pi[k]) + np.sum(X * np.log(X_neg_k + EPS),
                                                     axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S[k] = (np.random.rand(self.S.shape[1]) < ratio)

    def _gibbs_sample_WH(self, X):
        X_hat = self.W.dot(self.H * self.S) + EPS
        # update variational parameters for components W
        a_W = self.a + self.W * (X / X_hat).dot((self.H * self.S).T)
        b_W = self.b + (self.H * self.S).sum(axis=1)
        self.W = np.random.gamma(a_W, 1. / b_W)

        # update variational parameters for activations H
        c_H = self.c + self.H * self.S * self.W.T.dot(X / X_hat)
        d_H = self.d + self.W.sum(axis=0, keepdims=True).T * self.S
        self.H = np.random.gamma(c_H, 1. / d_H)

        # update variational parameters for sparsity pi
        a_pi = self.a0 / self.n_components + self.S.sum(axis=1)
        b_pi = self.b0 * (self.n_components - 1) / self.n_components \
            + self.S.shape[1] - self.S.sum(axis=1)
        self.pi = np.random.beta(a_pi, b_pi)

    def _log_likelihood(self, X):
        log_ll = 0.
        log_ll += scipy.stats.gamma.logpdf(self.W, self.a, scale=1. / self.b).sum()
        log_ll += scipy.stats.gamma.logpdf(self.H, self.c, scale=1. / self.d).sum()
        # avoid inf log-likeli
        safe_pi = np.maximum(self.pi, EPS)
        safe_pi = np.minimum(safe_pi, 1 - EPS)
        log_ll += scipy.stats.beta.logpdf(safe_pi, self.a0 / self.n_components,
                                          self.b0 * (self.n_components - 1) / self.n_components).sum()
        log_ll += scipy.stats.bernoulli.logpmf(self.S, self.pi[:, np.newaxis]).sum()
        log_ll += scipy.stats.poisson.logpmf(X, (self.W.dot(self.H * self.S) + EPS)).sum()
        return log_ll

    def transform(self, X):
        raise NotImplementedError('Wait for it')
