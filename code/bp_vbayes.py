#!/usr/bin/env python
"""Beta-Process Sparse NMF

Two different approaches included for variational inference:
    BP_NMF --> Directly optimizing the ELBO for nonconjugate variables
    LVI_BP_NMF --> Laplace approximation variational inference

The algorithm is described in:
    'Beta Process Sparse Nonnegative Matrix Factorization for Music' by  
    Dawen Liang, Matthew D. Hoffman, and Daniel P. W. Ellis
    in ISMIR 2013

CREATED: 2013-04-24 18:14:24 by Dawen Liang <dl2771@columbia.edu>    

"""

import sys, math
import numpy as np
import scipy.optimize as optimize
import scipy.special as special

class BP_NMF:
    def __init__(self, X, K=512, smoothness=100, seed=None, **kwargs):
        '''
        BN = Bp_NMF(X, K=512, smoothness=100, seed=None, alpha=2., a0=1., b0=1., c0=1e-6, d0=1e-6)

        Required arguments:
            X:              F-by-T nonnegative matrix (numpy.ndarray) 
                            the data to be factorized

        Optional arguments:
            K:              the size of the initial dictionary
                            will be truncated to a proper size
            
            smoothness:     control the concentration of the variational 
                            parameters

            seed:           the random seed to control the random behavior

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
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self._init(smoothness)

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

    def _init(self, smoothness):
        # variational parameters for D (Phi)
        self.mu_phi = np.random.randn(self.F, self.K)
        self.r_phi = np.random.gamma(2, size=(self.F, self.K))
        self.sigma_phi = np.sqrt(1./self.r_phi)
        self.ED, self.ED2, _ = _exp(self.mu_phi, self.r_phi)
        # variational parameters for S (Psi)
        self.mu_psi = np.random.randn(self.K, self.T)
        self.r_psi = np.random.gamma(2, size=(self.K, self.T))
        self.sigma_psi = np.sqrt(1./self.r_psi)
        self.ES, self.ES2, self.ESinv = _exp(self.mu_psi, self.r_psi)
        # variational parameters for Z
        self.p_z = np.random.rand(self.K, self.T)
        self.EZ = self.p_z
        # variational parameters for pi
        self.alpha_pi = np.random.rand(self.K)
        self.beta_pi = np.random.rand(self.K)
        self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)
        # variational parameters for gamma
        self.alpha_g, self.beta_g = np.random.gamma(smoothness, 1./smoothness), np.random.gamma(smoothness, 1./smoothness)
        self.Eg = self.alpha_g / self.beta_g
        self.good_k = np.arange(self.K)

    def update(self, verbose=True, disp=0):
        '''
        Perform dictionary-learning update for one iteration, truncate rarely-used dictionary elements and update the lower bound.  

        return True if sucessfully completed, False otherwise
        '''
        print 'Updating DZS...'
        good_k = self.good_k
        for k in good_k:
            self.update_phi(k, disp)
            self.update_z(k)
            self.update_psi(k, disp)
            if verbose and not k % 5:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating pi and gamma...'
        self.update_pi()
        self.update_r()
        # truncate the rarely used elements
        self.good_k = np.delete(good_k, np.where(self.Epi[good_k] < 1e-3*np.max(self.Epi[good_k])))
        self._lower_bound()
        return True 

    def update_phi(self, k, disp):
        def f(phi):
            pass

        def df(phi):
            pass
        pass

    def update_psi(self, k, disp):
        def f(psi):
            pass
        def df(psi):
            pass
        pass
    
    def update_z(self, k):
        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        dummy = self.Eg * (-1./2 * np.outer(self.ED2[:,k], self.ES2[k,:]).sum(axis=0) + np.sum(np.outer(self.ED[:,k], self.ES[k,:]) * Eres, axis=0))
        p0 = special.psi(self.beta_pi[k]) - special.psi(self.alpha_pi[k] + self.beta_pi[k])
        p1 = special.psi(self.alpha_pi[k]) - special.psi(self.alpha_pi[k] + self.beta_pi[k]) + dummy
        self.p_z[k,:] = 1./(1 + np.exp(p0 - p1))
        self.EZ[k,:] = self.p_z[k,:]

    def update_pi(self):
        self.alpha_pi = self.a0/self.K + np.sum(self.EZ, axis=1)
        self.beta_pi = self.b0 * (self.K-1) / self.K + self.T - np.sum(self.EZ, axis=1)
        self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)

    def update_r(self):
        good_k = self.good_k
        self.alpha_g = self.c0 + 1./2 * self.F * self.T
        self.beta_g = self.d0 + 1./2 * np.sum((self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:] * self.EZ[good_k,:]))**2)
        self.Eg = self.alpha_g / self.beta_g

    def _lower_bound(self):
        # E[log P(X | Phi, Psi, Z, gamma)]
        Xres = np.dot(self.ED, self.ES*self.EZ)
        Xres2 = np.dot(self.ED2, self.ES2*self.EZ)
        EetaX =  self.Eg * (self.X * Xres - 1./2 * self.X**2)
        EAeta = 1./2 * self.Eg * (Xres2 + (Xres**2 - np.dot(self.ED**2, self.ES**2 * self.EZ))) - 1./2 * (special.psi(self.alpha_g) - np.log(self.beta_g))
        self.obj = np.sum(EetaX - EAeta) 
        # E[log P(Phi) - log q(Phi)]
        self.obj += -1./2 * np.sum(self.mu_phi**2)  # E[log P(Phi)]
        idx_phi = np.isinf(self.r_phi)
        self.obj += np.sum(np.log(2*math.pi*math.e/self.r_phi[~idx_phi]))/2 # E[log q(Phi)]
        # E[log P(Psi) - log q(Psi)]
        self.obj += np.sum(self.alpha * self.mu_psi - self.alpha * np.exp(self.mu_psi + 1./(2*self.r_psi)))
        idx_psi = np.isinf(self.r_psi)
        self.obj += np.sum(np.log(2*math.pi*math.e/self.r_psi[~idx_psi]))/2
        # E[log P(Z | pi) - log q(Z)]
        idx_pi = (self.Epi != 0) & (self.Epi != 1)
        idx_pz = (self.p_z != 0) & (self.p_z != 1)
        self.obj += self.T * np.sum(self.Epi[idx_pi] * np.log(self.Epi[idx_pi]) + (1-self.Epi[idx_pi]) * np.log(1-self.Epi[idx_pi])) + np.sum(-self.p_z[idx_pz] * np.log(self.p_z[idx_pz]) - (1-self.p_z[idx_pz]) * np.log(1-self.p_z[idx_pz])) 
        # E[log P(pi) - log q(pi)]
        tmp_alpha, tmp_beta = self.a0/self.K, self.b0*(self.K-1)/self.K
        Elog_mpi = np.sum(special.psi(self.beta_pi) - special.psi(self.alpha_pi + self.beta_pi))
        Elog_pi = np.sum(special.psi(self.alpha_pi) - special.psi(self.alpha_pi + self.beta_pi))
        self.obj += (tmp_alpha - 1) * Elog_pi + (tmp_beta - 1) * Elog_mpi
        self.obj += np.sum(special.beta(self.alpha_pi, self.beta_pi) - (self.alpha_pi - 1) * special.psi(self.alpha_pi) - (self.beta_pi - 1) * special.psi(self.beta_pi) + (self.alpha_pi + self.beta_pi - 2) * special.psi(self.alpha_pi + self.beta_pi))
        # E[log P(gamma) - log q(gamma)]
        self.obj += (self.c0 - 1) * (special.psi(self.alpha_g) - np.log(self.beta_g)) - self.d0 * self.Eg     # E[log P(gamma)]
        self.obj += self.alpha_g - np.log(self.beta_g) + special.gammaln(self.alpha_g) + (1-self.alpha_g) * special.psi(self.alpha_g)



class LVI_BP_NMF(BP_NMF):
    def __init__(self, X, K=512, init_option='Rand', seed=None, **kwargs):
        super(LVI_BP_NMF, self).__init__(X, K=K, init_option=init_option, seed=seed, **kwargs)

    def update(self, verbose=True, disp=0):
        '''
        Perform dictionary-learning update for one iteration, truncate rarely-used dictionary elements and update the lower bound.  

        return True if sucessfully completed, False otherwise
        '''
        print 'Updating DZS...'
        good_k = self.good_k
        for k in good_k:
            ind_phi = self.update_phi(k, disp)
            self.update_z(k)
            ind_psi = self.update_psi(k, disp)
            if not ind_phi or not ind_psi:
                # LBFGS fucked up
                return False
            if verbose and not k % 5:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating pi and gamma...'
        self.update_pi()
        self.update_r()
        # truncate the rarely used elements
        self.good_k = np.delete(good_k, np.where(self.Epi[good_k] < 1e-3*np.max(self.Epi[good_k])))
        self._lower_bound()
        return True

    def update_phi(self, k, disp):
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

        def df2(phi, f=np.arange(self.F)):
            lcoef, qcoef = f_stub(phi, f)
            const = -1
            return -(lcoef + 4*qcoef + const)

        good_k = self.good_k
        Eres = self.X - np.dot(self.ED[:,good_k], self.ES[good_k,:]*self.EZ[good_k,:]) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        phi0 = self.mu_phi[:,k]
        mu_hat, _, d = optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=0)
        self.mu_phi[:,k], self.r_phi[:,k] = mu_hat, df2(mu_hat)
        if disp and np.any(self.r_phi[:,k] <= 0):
            if d['warnflag'] == 2:
                print 'D[:, {}]: {}, f={}'.format(k, d['task'], f(mu_hat))
            else:
                print 'D[:, {}]: {}, f={}'.format(k, d['warnflag'], f(mu_hat))
            return False 
        self.ED[:,k], self.ED2[:,k], _ = _exp(self.mu_phi[:,k], self.r_phi[:,k])
        return True 
  
    def update_psi(self, k, disp):
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
        mu_hat, _, d = optimize.fmin_l_bfgs_b(f, psi0, fprime=df, disp=0)
        self.mu_psi[k,:], self.r_psi[k,:] = mu_hat, df2(mu_hat)
        if np.any(self.r_psi[k, :] <= 0):
            if d['warnflag'] == 2:
                print 'S[{}, :]: {}'.format(k, d['task'])
            else:
                print 'S[{}, :]: {}'.format(k, d['warnflag'])
            return False 
        self.ES[k,:], self.ES2[k,:], self.ESinv[k,:] = _exp(self.mu_psi[k,:], self.r_psi[k,:])
        return True 
    
def _exp(mu, r):
    '''
    Given mean and precision of a Gaussian r.v. theta ~ N(mu, 1/r), compute E[exp(theta)], E[exp(2*theta)], and E[exp(-theta)]
    '''
    return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), np.exp(-mu + 1./(2*r)))


