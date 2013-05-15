#!/usr/bin/env python
'''Bayesian nonparametric NMF with Laplace Approximation Variational Inference

2013-04-24 18:14:24 by Dawen Liang <dl2771@columbia.edu>    

'''

import sys, math
import numpy as np
import scipy.optimize, scipy.special

class Bp_NMF:
    def __init__(self, X, K=512, init_option='Rand', encoding=False, RSeed=np.random.seed(), **kwargs):
        '''
        BN = Bp_NMF(X, K=512, init_option='Rand', encoding=False, RSeed=np.random.seed(), alpha=2., a0=1., b0=1., c0=1e-6, d0=1e-6)

        Required arguments:
            X:              F-by-T nonnegative matrix (numpy.ndarray) 
                            the data to be factorized

        Optional arguments:
            K:              the size of the initial dictionary
                            will be truncated to a proper size
            
            init_option:    must be 'Rand' or 'NMF'
                            The initialization rule to start the inference
                            'Rand' will init all the parameters randomly
                            'NMF' will perform a regular NMF and init from there

            RSeed:          the random seed to control the random behavior

            encoding:       indicate if for dictionary learning or encoding
                            if True, even dictionary will not be updated

            alpha:          hyperparameter for activation.

            a0, b0:         both must be specified
                            hyperparameters for sparsity

            c0, d0:         both must be specified
                            hyperparameters for Gaussian noise

        '''
        self.X = X.copy()
        self.F, self.T = self.X.shape
        self.K = K
        self._parse_args(**kwargs)
        self._init(init_option, encoding)

    def _parse_args(self, **kwargs):
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 2.

        if 'a0' in kwargs and 'b0' in kwargs:
            self.a0, self.b0 = kwargs['a0'], kwargs['b0']
        else:
            self.a0, self.b0 = 1., 1.

        if 'c0' in kwargs and 'd0' in kwargs:
            self.c0, self.d0 = kwargs['c0'], kwargs['d0']
        else:
            self.c0, self.d0 = 1e-6, 1e-6

    def _init(self, init_option, encoding):
        if init_option == 'Rand':
            print 'Init with Rand...'
            if ~encoding:
                # variational parameters for D (Phi)
                self.mu_phi = np.random.randn(self.F, self.K)
                self.r_phi = np.random.gamma(2, size=(self.F, self.K))
                self.ED, self.ED2, _ = self._exp(self.mu_phi, self.r_phi)
            # variational parameters for S (Psi)
            self.mu_psi = np.random.randn(self.K, self.T)
            self.r_psi = np.random.gamma(2, size=(self.K, self.T))
            self.ES, self.ES2, self.ESinv = self._exp(self.mu_psi, self.r_psi)
            # variational parameters for Z
            self.p_z = np.random.rand(self.K, self.T)
            self.EZ = self.p_z
            # variational parameters for pi
            self.alpha_pi = np.random.rand(self.K)
            self.beta_pi = np.random.rand(self.K)
            self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)
            # variational parameters for gamma
            self.alpha_g, self.beta_g = np.random.gamma(100, 1./100), np.random.gamma(100, 1./100)
            self.Eg = self.alpha_g / self.beta_g

        if init_option == 'NMF':
            try:
                import pymf
            except ImportError:
                print 'No pymf module, init with random...'
                self._init('Rand', encoding)
            n_basis = min(self.K, self.T, 100) 
            print 'Init with NMF ({} basis)...'.format(n_basis)
            nmf = pymf.NMF(self.X, num_bases=n_basis, niter=100)
            nmf.initialization()
            nmf.factorize()
            if ~encoding:
                # variational parameter for D (Phi)
                self.ED, self.ED2 = np.zeros((self.F, self.K)), np.zeros((self.F, self.K))
                self.ED[:,:n_basis], self.ED2[:,:n_basis] = nmf.W, nmf.W**2
                self.mu_phi, self.r_phi = np.empty((self.F, self.K)), np.empty((self.F, self.K))
                self.mu_phi[:,:n_basis] = np.log(nmf.W)
            # variational parameter for S (Psi)
            self.ES, self.ES2, self.ESinv = np.zeros((self.K, self.T)), np.zeros((self.K, self.T)), np.zeros((self.K, self.T))
            self.ES[:n_basis,:], self.ES2[:n_basis,:], self.ESinv[:n_basis,:] = nmf.H, nmf.H**2, 1./nmf.H
            self.mu_psi, self.r_psi = np.empty((self.K, self.T)), np.empty((self.K, self.T))
            self.mu_psi[:n_basis,:] = np.log(nmf.H)
            # variational parameter for Z
            self.p_z = np.zeros((self.K, self.T))
            self.p_z[:n_basis,:] = 1
            self.EZ = self.p_z
            # variational parameter for pi
            self.alpha_pi = np.random.rand(self.K)
            self.beta_pi = np.random.rand(self.K)
            self.Epi = np.zeros((self.K,))
            self.Epi[:self.T] = 1
            # variational parameter for gamma
            self.alpha_g, self.beta_g = np.random.gamma(100, 1./1000), np.random.gamma(100, 1./1000)
            self.Eg = 1./np.var(self.X - np.dot(self.ED, self.ES*self.EZ))
        self.good_k = np.arange(self.K)

    def _exp(self, mu, r):
        '''
        Given mean and precision of a Gaussian r.v. theta ~ N(mu, 1/r), compute E[exp(theta)], E[exp(2*theta)], and E[exp(-theta)]
        '''
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), np.exp(-mu + 1./(2*r)))

    def update(self, timed=False, verbose=True):
        '''
        Perform update for one iteration, truncate unimportant dictionary elements and update the lower bound.  
        '''
        print 'Updating DZS...'
        good_k = self.good_k
        for k in good_k:
            ind_phi = self.update_phi(k)
            self.update_z(k)
            ind_psi = self.update_psi(k, timed=timed)
            if ind_phi == -1 or ind_psi == -1:
                # something fucked up
                return -1
            if verbose and k % 5 == 0:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating pi and gamma...'
        self.update_pi()
        self.update_r()
        # truncate the rarely used elements
        self.good_k = np.delete(good_k, np.where(self.Epi[good_k] < 1e-3*np.max(self.Epi[good_k])))
        self._lower_bound(timed=timed)
        return 0

    def encode(self, timed=False, verbose=True):
        print 'Updating ZS...'
        good_k = self.good_k
        for k in good_k:
            self.update_z(k)
            ind_psi = self.update_psi(k, timed=timed)
            if ind_psi == -1:
                return -1
            if verbose and k % 5 == 0:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Update pi and gamma...'
        self.update_pi()
        self.update_r()
        self._lower_bound(timed=timed)
        return 0
        
    def update_phi(self, k):
        def f_stub(phi):
            lcoef = self.Eg * np.sum(np.outer(np.exp(phi), self.ES[k,:]*self.EZ[k,:]) * Eres, axis=1)
            qcoef = -1./2 * self.Eg * np.sum(np.outer(np.exp(2*phi), self.ES2[k,:]*self.EZ[k,:]), axis=1)
            return (lcoef, qcoef)

        def f(phi):
            lcoef, qcoef = f_stub(phi)
            const = -1./2*phi**2
            return -np.sum(lcoef + qcoef + const)

        def df(phi):
            lcoef, qcoef = f_stub(phi)
            const = -phi
            return -(lcoef + 2*qcoef + const)

        def df2(phi):
            lcoef, qcoef = f_stub(phi)
            const = -1
            return -(lcoef + 4*qcoef + const)

        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        phi0 = self.mu_phi[:,k]
        mu_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=0)
        self.mu_phi[:,k], self.r_phi[:,k] = mu_hat, df2(mu_hat)
        if np.alltrue(self.r_phi[:,k] > 0) == False:
            if d['warnflag'] == 2:
                print 'D[:, {}]: {}, f={}'.format(k, d['task'], f(mu_hat))
            else:
                print 'D[:, {}]: {}, f={}'.format(k, d['warnflag'], f(mu_hat))
            if np.isnan(f(mu_hat)):
                print 'ED no NAN: {}'.format(np.alltrue(~np.isnan(self.ED[:,good_k])))
                print 'ES no NAN: {}'.format(np.alltrue(~np.isnan(self.ES[good_k,:])))
                print 'EZ no NAN: {}'.format(np.alltrue(~np.isnan(self.EZ[good_k,:])))
                print 'EX no NAN: {}'.format(np.alltrue(~np.isnan(np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]))))
            return -1
        self.ED[:,k], self.ED2[:,k], _ = self._exp(self.mu_phi[:,k], self.r_phi[:,k])
        return 0
  
    def update_z(self, k):
        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        dummy = self.Eg * (-1./2 * np.outer(self.ED2[:,k], self.ES2[k,:]).sum(axis=0) + np.sum(np.outer(self.ED[:,k], self.ES[k,:]) * Eres, axis=0))
        p0 = scipy.special.psi(self.beta_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k])
        p1 = scipy.special.psi(self.alpha_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k]) + dummy
        self.p_z[k,:] = 1./(1 + np.exp(p0 - p1))
        self.EZ[k,:] = self.p_z[k,:]

    def update_psi(self, k, timed=False):
        if timed:
            return self._update_psi_time(k)
        else:
            return self._update_psi_ntime(k)

    def _update_psi_ntime(self, k):
        def f_stub(psi):
            lcoef = self.Eg * np.sum(np.outer(self.ED[:,k], np.exp(psi)*self.EZ[k,:]) * Eres, axis=0)
            qcoef = -1./2 * self.Eg * np.sum(np.outer(self.ED2[:,k], np.exp(2*psi)*self.EZ[k,:]), axis=0)
            return (lcoef, qcoef)

        def f(psi):
            lcoef, qcoef = f_stub(psi)
            const = self.alpha * psi - self.alpha * np.exp(psi)
            return -np.sum(lcoef + qcoef + const)
        def df(psi):
            lcoef, qcoef = f_stub(psi)
            const = self.alpha - self.alpha * np.exp(psi)
            return -(lcoef + 2*qcoef + const)
            
        def df2(psi):
            lcoef, qcoef = f_stub(psi)
            const = -self.alpha * np.exp(psi)
            return -(lcoef + 4*qcoef + const)

        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        psi0 = self.mu_psi[k,:]
        mu_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, psi0, fprime=df, disp=0)
        self.mu_psi[k,:], self.r_psi[k,:] = mu_hat, df2(mu_hat)
        if np.alltrue(self.r_psi[k,:] > 0) == False:
            if d['warnflag'] == 2:
                print 'S[{}, :]: {}'.format(k, d['task'])
            else:
                print 'S[{}, :]: {}'.format(k, d['warnflag'])
            return -1
        self.ES[k,:], self.ES2[k,:], self.ESinv[k,:] = self._exp(self.mu_psi[k,:], self.r_psi[k,:])
        return 0

    def _update_psi_time(self, k):
        def f_stub(psi):
            lcoef = self.Eg * np.sum(np.outer(self.ED[:,k], np.exp(psi) * self.EZ[k,ts]) * Eres[:,ts], axis=0)
            qcoef = -1./2 * self.Eg * np.sum(np.outer(self.ED2[:,k], np.exp(2*psi) * self.EZ[k,ts]), axis=0)
            return (lcoef, qcoef)
        
        def f(psi):
            lcoef, qcoef = f_stub(psi)
            lt = ts.shape[0]
            bwd = np.empty((lt,))
            fwd = np.empty((lt,))
            if ts[0] == 0:
                bwd[0] = self.alpha * psi[0] - self.alpha * np.exp(psi[0])
                bwd[1:] = self.alpha * psi[1:] - self.alpha * self.ESinv[k, ts[1:]-1] * np.exp(psi[1:])
            else:
                bwd = self.alpha * psi - self.alpha * self.ESinv[k, ts-1] * np.exp(psi)
            if ts[-1] == self.T-1:
                fwd[-1] = 0
                fwd[:-1] = -self.alpha * self.ES[k, ts[:-1]+1] * np.exp(-psi[:-1]) - self.alpha * psi[:-1]
            else:
                fwd = -self.alpha * self.ES[k, ts+1] * np.exp(-psi) - self.alpha * psi
            return -np.sum(lcoef + qcoef + bwd + fwd)
       
        def df(psi):
            lcoef, qcoef = f_stub(psi)
            lt = ts.shape[0]
            bwd = np.empty((lt,))
            fwd = np.empty((lt,))
            if ts[0] == 0:
                bwd[0] = self.alpha - self.alpha * np.exp(psi[0])
                bwd[1:] = self.alpha - self.alpha * self.ESinv[k, ts[1:]-1] * np.exp(psi[1:])
            else:
                bwd = self.alpha - self.alpha * self.ESinv[k, ts-1] * np.exp(psi)
            if ts[-1] == self.T-1:
                fwd[-1] = 0
                fwd[:-1] = self.alpha * self.ES[k, ts[:-1]+1] * np.exp(-psi[:-1]) - self.alpha
            else:
                fwd = self.alpha * self.ES[k, ts+1] * np.exp(-psi) - self.alpha
            return -(lcoef + 2*qcoef + bwd + fwd)
        
        def df2(psi):
            lcoef, qcoef = f_stub(psi)
            lt = ts.shape[0]
            bwd = np.empty((lt,))
            fwd = np.empty((lt,))
            if ts[0] == 0:
                bwd[0] = -self.alpha * np.exp(psi[0])
                bwd[1:] = -self.alpha * self.ESinv[k, ts[1:]-1] * np.exp(psi[1:])
            else:
                bwd = -self.alpha * self.ESinv[k, ts-1] * np.exp(psi)
            if ts[-1] == self.T-1:
                fwd[-1] = 0
                fwd[:-1] = -self.alpha * self.ES[k, ts[:-1]+1] * np.exp(-psi[:-1])
            else:
                fwd = -self.alpha * self.ES[k, ts+1] * np.exp(-psi) 
            return -(lcoef + 4*qcoef + bwd + fwd)

        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        for st in xrange(2):
            ts = np.arange(st, self.T, 2)
            mu_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, self.mu_psi[k, ts], fprime=df, disp=0)
            self.mu_psi[k, ts], self.r_psi[k, ts] = mu_hat, df2(mu_hat)
            if ~np.alltrue(self.r_psi[k,ts] > 0):
                if d['warnflag'] == 2:
                    print 'S[{}, :]:{}, f={}'.format(k, d['task'], f(mu_hat))
                else:
                    print 'S[{}, :]:{}, f={}'.format(k, d['warnflag'], f(mu_hat))
                return -1
            self.ES[k,ts], self.ES2[k,ts], self.ESinv[k,ts] = self._exp(self.mu_psi[k,ts], self.r_psi[k,ts])
            if ~np.alltrue(~np.isinf(self.ES[k,ts])):
                print 'Inf ES'
                return -1
            if ~np.alltrue(~np.isinf(self.ES2[k,ts])):
                print 'Inf ES2'
                return -1
            if ~np.alltrue(~np.isinf(self.ESinv[k,ts])):
                print 'Inf ESinv'
                return -1
            return 0

    def update_pi(self):
        self.alpha_pi = self.a0/self.K + np.sum(self.EZ, axis=1)
        self.beta_pi = self.b0 * (self.K-1) / self.K + self.T - np.sum(self.EZ, axis=1)
        self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)

    def update_r(self):
        good_k = self.good_k
        self.alpha_g = self.c0 + 1./2 * self.F * self.T
        self.beta_g = self.d0 + 1./2 * np.sum((self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:] * self.EZ[good_k,:]))**2)
        self.Eg = self.alpha_g / self.beta_g

    def _lower_bound(self, timed=False):
        # E[log P(X | Phi, Psi, Z, gamma)]
        Xres = np.dot(self.ED, self.ES*self.EZ)
        Xres2 = np.dot(self.ED2, self.ES2*self.EZ)
        EetaX =  self.Eg * (self.X * Xres - 1./2 * self.X**2)
        EAeta = 1./2 * self.Eg * (Xres2 + (Xres**2 - np.dot(self.ED**2, self.ES**2 * self.EZ))) - 1./2 * (scipy.special.psi(self.alpha_g) - np.log(self.beta_g))
        self.obj = np.sum(EetaX - EAeta) 
        # E[log P(Phi) - log q(Phi)]
        self.obj += -1./2 * np.sum(self.mu_phi**2)  # E[log P(Phi)]
        idx_phi = np.isinf(self.r_phi)
        self.obj += np.sum(np.log(2*math.pi*math.e/self.r_phi[~idx_phi]))/2 # E[log q(Phi)]
        # E[log P(Psi) - log q(Psi)]
        if timed:
            tmp = np.zeros((self.K, self.T))
            tmp[:,0] = self.alpha * self.mu_psi[:,0] - self.alpha * np.exp(self.mu_psi[:,0] + 1./(2*self.r_psi[:,0]))
            tmp[:,1:] = self.alpha * self.mu_psi[:,1:] - self.alpha * np.exp(-self.mu_psi[:,:-1] + 1./(2*self.r_psi[:,:-1])) * np.exp(self.mu_psi[:,1:] + 1./(2*self.r_psi[:,1:]))
            self.obj += np.sum(tmp)
        else:
            self.obj += np.sum(self.alpha * self.mu_psi - self.alpha * np.exp(self.mu_psi + 1./(2*self.r_psi)))
        idx_psi = np.isinf(self.r_psi)
        self.obj += np.sum(np.log(2*math.pi*math.e/self.r_psi[~idx_psi]))/2
        # E[log P(Z | pi) - log q(Z)]
        idx_pi = (self.Epi != 0) & (self.Epi != 1)
        idx_pz = (self.p_z != 0) & (self.p_z != 1)
        self.obj += self.T * np.sum(self.Epi[idx_pi] * np.log(self.Epi[idx_pi]) + (1-self.Epi[idx_pi]) * np.log(1-self.Epi[idx_pi])) + np.sum(-self.p_z[idx_pz] * np.log(self.p_z[idx_pz]) - (1-self.p_z[idx_pz]) * np.log(1-self.p_z[idx_pz])) 
        # E[log P(pi) - log q(pi)]
        tmp_alpha, tmp_beta = self.a0/self.K, self.b0*(self.K-1)/self.K
        Elog_mpi = np.sum(scipy.special.psi(self.beta_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi))
        Elog_pi = np.sum(scipy.special.psi(self.alpha_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi))
        self.obj += (tmp_alpha - 1) * Elog_pi + (tmp_beta - 1) * Elog_mpi
        self.obj += np.sum(scipy.special.beta(self.alpha_pi, self.beta_pi) - (self.alpha_pi - 1) * scipy.special.psi(self.alpha_pi) - (self.beta_pi - 1) * scipy.special.psi(self.beta_pi) + (self.alpha_pi + self.beta_pi - 2) * scipy.special.psi(self.alpha_pi + self.beta_pi))
        # E[log P(gamma) - log q(gamma)]
        self.obj += (self.c0 - 1) * (scipy.special.psi(self.alpha_g) - np.log(self.beta_g)) - self.d0 * self.Eg     # E[log P(gamma)]
        ## Stirling to approximate log(gamma(alpha_g))
        self.obj += self.alpha_g - np.log(self.beta_g) + ((self.alpha_g-.5)*np.log(self.alpha_g) - self.alpha_g + .5*np.log(2*math.pi)) + (1-self.alpha_g) * scipy.special.psi(self.alpha_g)
        pass
