#!/usr/bin/env python
"""
Beta-process NMF with stochastic structured mean-field variational inference

CREATED: 2014-10-14 15:38:10 by Dawen Liang <dliang@ee.columbia.edu>

"""

import numpy as np
import scipy.stats

import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)


class SSMF_BP_NMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=500, max_iter=100, burn_in=1000,
                 cutoff=1e-3, smoothness=100, random_state=None,
                 verbose=False, **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.burn_in = burn_in
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

        # hyperparameters for stochastic (natural) gradient
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.5))

    def _init_components(self, n_feats):
        # variational parameters for components W
        self.nu_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))
        self.rho_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))

        # variational parameters for sparsity pi
        self.alpha_pi = np.random.rand(self.n_components)
        self.beta_pi = np.random.rand(self.n_components)

    def _init_weights(self, n_samples):
        # variational parameters for activations H
        self.nu_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))
        self.rho_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))

    def fit(self, X):
        n_feats, n_samples = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self.good_k = np.arange(self.n_components)
        # randomly initalize binary mask
        self.S = (np.random.rand(self.n_components, n_samples) > .5)
        self._ssmf_a(X)

        return self

    def _ssmf_a(self, X):
        self.log_ll = np.zeros((self.max_iter, self.burn_in+1))
        for i in xrange(self.max_iter):
            good_k = self.good_k
            if self.verbose:
                print 'SSMF-A iteration %d\tgood K:%d:' % (i, good_k.size)
                sys.stdout.flush()
            eta = (self.t0 + i)**(-self.kappa)
            W = np.random.gamma(self.nu_W[:, good_k],
                                1. / self.rho_W[:, good_k])
            H = np.random.gamma(self.nu_H[good_k], 1. / self.rho_H[good_k])
            pi = np.random.beta(self.alpha_pi[good_k], self.beta_pi[good_k])
            for b in xrange(self.burn_in+1):
                # burn-in plus one actual sample
                self.gibbs_sample_S(X, W, H, pi)
                self.log_ll[i, b] = _log_likelihood(X, self.S[good_k], W, H,
                                                    pi)
                if self.verbose and b % 10 == 0:
                    sys.stdout.write('\r\tGibbs burn-in: %d' % b)
                    sys.stdout.flush()
            if self.verbose:
                sys.stdout.write('\n')
            self._update(eta, X, W, H)

            Epi = self.alpha_pi[good_k] / (self.alpha_pi[good_k] +
                                           self.beta_pi[good_k])
            self.good_k = good_k[Epi > Epi.max() * self.cutoff]
        pass

    def gibbs_sample_S(self, X, W, H, pi, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            X_neg_k = W.dot(H * self.S[good_k]) - np.outer(W[:, i],
                                                           H[i] * self.S[k])
            log_Ph = np.log(pi[i]) + np.sum(X * np.log(X_neg_k +
                                                       np.outer(W[:, i], H[i]))
                                            - np.outer(W[:, i], H[i]), axis=0)
            log_Pt = np.log(1 - pi[i]) + np.sum(X * np.log(X_neg_k + EPS),
                                                axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S[k] = (np.random.rand(self.S.shape[1]) < ratio)
            if type(log_ll) is list:
                log_ll.append(_log_likelihood(X, self.S[good_k], W, H, pi))
        pass

    def _update(self, eta, X, W, H):
        good_k = self.good_k
        X_hat = W.dot(H * self.S[good_k]) + EPS
        # update variational parameters for components W
        self.nu_W[:, good_k] = (1 - eta) * self.nu_W[:, good_k] + \
            eta * (self.a + W * (X / X_hat).dot((H * self.S[good_k]).T))
        self.rho_W[:, good_k] = (1 - eta) * self.rho_W[:, good_k] + \
            eta * (self.b + H.sum(axis=1))

        # update variational parameters for activations H
        self.nu_H[good_k] = (1 - eta) * self.nu_H[good_k] + \
            eta * (self.c + H * self.S[good_k] * W.T.dot(X / X_hat))
        self.rho_H[good_k] = (1 - eta) * self.rho_H[good_k] + \
            eta * (self.d + W.sum(axis=0)[:, np.newaxis])

        # update variational parameters for sparsity pi
        self.alpha_pi[good_k] = (1 - eta) * self.alpha_pi[good_k] + \
            eta * (self.a0 / self.n_components + self.S[good_k].sum(axis=1))
        self.beta_pi[good_k] = (1 - eta) * self.beta_pi[good_k] + \
            eta * (self.b0 * (self.n_components - 1) / self.n_components
                   + self.S.shape[1] - self.S[good_k].sum(axis=1))

    def transform(self, X):
        raise NotImplementedError('Wait for it')


def _log_likelihood(X, S, W, H, pi):
    log_ll = scipy.stats.bernoulli.logpmf(S, pi[:, np.newaxis]).sum()
    log_ll += scipy.stats.poisson.logpmf(X, (W.dot(H * S) + EPS)).sum()

    return log_ll
