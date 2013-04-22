import sys, math
import numpy as np
import scipy.optimize, scipy.special

class Bp_NMF:
    def __init__(self, X, K=512, RSeed=np.random.seed(), **kwargs):
        self.X = X.copy()
        self.F, self.T = self.X.shape
        self.K = K
        self._parse_args(**kwargs)
        self._init()

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

    def _init(self, init_option='Rand', num_bases=100):
        if init_option == 'Rand':
            print 'Init with Rand...'
            # variational parameters for D (Phi)
            self.mu_phi = np.random.randn(self.F, self.K)
            self.r_phi = np.random.gamma(2, size=(self.F, self.K))
            # variational parameters for S (Psi)
            self.mu_psi = np.random.randn(self.K, self.T)
            self.r_psi = np.random.gamma(2, size=(self.K, self.T))
            # variational parameters for Z
            self.p_z = np.random.rand(self.K, self.T)
            # variational parameters for pi
            self.alpha_pi = np.random.rand(self.K)
            self.beta_pi = np.random.rand(self.K)
            # variational parameters for gamma
            self.alpha_g, self.beta_g = np.random.gamma(100, 1./100), np.random.gamma(100, 1./100)

            # init the expectations
            self.ED, self.ED2, _ = self._exp(self.mu_phi, self.r_phi)
            self.ES, self.ES2, self.ESinv = self._exp(self.mu_psi, self.r_psi)
            self.EZ = self.p_z
            self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)
            self.Eg = self.alpha_g / self.beta_g

    def _exp(self, mu, r):
        '''
        Given mean and precision of a Gaussian r.v. theta ~ N(mu, 1/r), compute E[exp(theta)], E[exp(2*theta)], and E[exp(-theta)]
        '''
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), np.exp(-mu + 1./(2*r)))

    def update(self, timed=False, verbose=True):
        print 'Updating DZS...'
        for k in xrange(self.K):
            self.update_phi(k)
            self.update_z(k)
            self.update_psi(k, timed)
            if verbose:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating pi and gamma...'
        self.update_pi()
        self.update_r()
        self._lower_bound()
        
    def update_phi(self, k):
        def f_stub(phi):
            lcoef = self.Eg * np.sum(np.outer(np.exp(phi), self.ES[k,:]*self.EZ[k,:]) * Eres, axis=1)
            qcoef = -1./2 * self.Eg * np.sum(np.outer(np.exp(2*phi), self.ES2[k,:]*self.EZ[k,:]), axis=1)
            return (lcoef, qcoef)

        def f(phi):
            lcoef, qcoef = f_stub(phi)
            const = -1./2*phi**2
            #if ~np.alltrue(~np.isnan(lcoef)):
            #    print 'LC: {}'.format(lcoef)
            #if ~np.alltrue(~np.isnan(qcoef)):
            #    print 'QC: {}'.format(qcoef)
            #if ~np.alltrue(~np.isnan(const)):
            #    print 'Const: {}'.format(const)
            return -np.sum(lcoef + qcoef + const)

        def df(phi):
            lcoef, qcoef = f_stub(phi)
            const = -phi
            return -(lcoef + 2*qcoef + const)

        def df2(phi):
            lcoef, qcoef = f_stub(phi)
            const = -1
            return -(lcoef + 4*qcoef + const)

        Eres = self.X - np.dot(self.ED, self.ES*self.EZ) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        phi0 = self.mu_phi[:,k]
        mu_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=5)
        self.mu_phi[:,k], self.r_phi[:,k] = mu_hat, df2(mu_hat)
        if np.alltrue(self.r_phi[:,k] > 0) == False:
            if d['warnflag'] == 2:
                print 'D[:, {}]: {}, f={}'.format(k, d['task'], f(mu_hat))
            else:
                print 'D[:, {}]: {}, f={}'.format(k, d['warnflag'], f(mu_hat))
            if np.isnan(f(mu_hat)):
                print np.alltrue(~np.isnan(self.ED))
                print np.alltrue(~np.isnan(self.ES))
                print np.alltrue(~np.isnan(self.EZ))
                print np.alltrue(~np.isnan(np.dot(self.ED, self.ES*self.EZ)))
        self.ED[:,k], self.ED2[:,k], _ = self._exp(self.mu_phi[:,k], self.r_phi[:,k])
  
    def update_z(self, k):
        Eres = self.X - np.dot(self.ED, self.ES*self.EZ) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        dummy = self.Eg * (-1./2 * np.outer(self.ED2[:,k], self.ES2[k,:]).sum(axis=0) + np.sum(np.outer(self.ED[:,k], self.ES[k,:]) * Eres, axis=0))
        p0 = scipy.special.psi(self.beta_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k])
        p1 = scipy.special.psi(self.alpha_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k]) + dummy
        self.p_z[k,:] = 1./(1 + np.exp(p0 - p1))
        self.EZ[k,:] = self.p_z[k,:]

    def update_psi(self, k, timed=False):
        if timed:
            self._update_psi_time(k)
        else:
            self._update_psi_ntime(k)

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

        Eres = self.X - np.dot(self.ED, self.ES*self.EZ) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        psi0 = self.mu_psi[k,:]
        mu_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, psi0, fprime=df, disp=5)
        self.mu_psi[k,:], self.r_psi[k,:] = mu_hat, df2(mu_hat)
        if np.alltrue(self.r_psi[k,:] > 0) == False:
            if d['warnflag'] == 2:
                print 'S[{}, :]: {}'.format(k, d['task'])
            else:
                print 'S[{}, :]: {}'.format(k, d['warnflag'])
        self.ES[k,:], self.ES2[k,:], self.ESinv[k,:] = self._exp(self.mu_psi[k,:], self.r_psi[k,:])

    def _update_psi_time(self, k):
        def f_stub(psi_t):
            lcoef = self.Eg * np.sum(self.ED[:,k] * np.exp(psi_t) * self.EZ[k,t] * Eres[:,t])
            qcoef = -1./2 * self.Eg * np.sum(self.ED2[:,k] * np.exp(2*psi_t) * self.EZ[k,t])
            return (lcoef, qcoef)

        def f(psi_t):
            lcoef, qcoef = f_stub(psi_t)
            if t == 0:
                bwd = self.alpha * psi_t - self.alpha * np.exp(psi_t)
            else:
                bwd = self.alpha * psi_t - self.alpha * self.ESinv[k,t-1] * np.exp(psi_t)
            if t == self.T-1:
                fwd = 0
            else:
                fwd = -self.alpha * self.ES[k, t+1] * np.exp(-psi_t) - self.alpha * psi_t
            const = bwd + fwd
            return -(lcoef + qcoef + const)

        def df(psi_t):
            lcoef, qcoef = f_stub(psi_t)
            if t == 0:
                bwd = self.alpha - self.alpha * np.exp(psi_t)
            else:
                bwd = self.alpha - self.alpha * self.ESinv[k,t-1] * np.exp(psi_t)
            if t == self.T-1:
                fwd = 0
            else:
                fwd = self.alpha * self.ES[k, t+1] * np.exp(-psi_t) - self.alpha 
            const = bwd + fwd
            return -(lcoef + 2*qcoef + const)

        def df2(psi_t):
            lcoef, qcoef = f_stub(psi_t)
            if t == 0:
                bwd = - self.alpha * np.exp(psi_t)
            else:
                bwd = - self.alpha * self.ESinv[k,t-1] * np.exp(psi_t)
            if t == self.T-1:
                fwd = 0
            else:
                fwd = -self.alpha * self.ES[k, t+1] * np.exp(-psi_t) 
            const = bwd + fwd
            return -(lcoef + 4*qcoef + const)
        
        Eres = self.X - np.dot(self.ED, self.ES*self.EZ) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:]) 
        for t in xrange(self.T):
            if self.EZ[k,t] == 0:
                self.ES[k,t], self.ES2[k,t], self.ESinv[k,t] = 1., 1., 1.
                continue
            mu_t_hat, _, d = scipy.optimize.fmin_l_bfgs_b(f, self.mu_psi[k,t], fprime=df, disp=5)
            self.mu_psi[k,t], self.r_psi[k,t] = mu_t_hat, df2(mu_t_hat)
            if self.r_psi[k,t] <= 0:
                if d['warnflag'] == 2:
                    print 'S[{}, {}]:{}, Z:{}, mu={}, f={}, df={}, df2={}'.format(k, t, d['task'], self.EZ[k,t], mu_t_hat, f(mu_t_hat), df(mu_t_hat), df2(mu_t_hat))
                else:
                    print 'S[{}, {}]:{}, Z:{}, mu={}, f={}, df={}, df2={}'.format(k, t, d['warnflag'], self.EZ[k,t], mu_t_hat, f(mu_t_hat), df(mu_t_hat), df2(mu_t_hat))
            self.ES[k,t], self.ES2[k,t], self.ESinv[k,t] = self._exp(self.mu_psi[k,t], self.r_psi[k,t])
            if np.isinf(self.ES[k,t]):
                print 'Inf ES'


    def update_pi(self):
        self.alpha_pi = self.a0/self.K + np.sum(self.EZ, axis=1)
        self.beta_pi = self.b0 * (self.K-1) / self.K + self.T - np.sum(self.EZ, axis=1)
        self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)

    def update_r(self):
        self.alpha_g = self.c0 + 1./2 * self.F * self.T
        self.beta_g = self.d0 + 1./2 * np.sum((self.X - np.dot(self.ED, self.ES * self.EZ))**2)
        self.Eg = self.alpha_g / self.beta_g

    def _lower_bound(self):
        # E[log P(X | Phi, Psi, Z, gamma)]
        Xres = np.dot(self.ED, self.ES*self.EZ)
        Xres2 = np.dot(self.ED2, self.ES2*self.EZ)
        EetaX =  self.Eg * (self.X * Xres - 1./2 * self.X**2)
        EAeta = 1./2 * self.Eg * (Xres2 + (Xres**2 - np.dot(self.ED**2, self.ES**2 * self.EZ))) - 1./2 * (scipy.special.psi(self.alpha_g) - np.log(self.beta_g))
        self.obj = np.sum(EetaX - EAeta) 
        # E[log P(Phi) - log q(Phi)], note that E[log P(phi)] is constant, thus is ignored
        self.obj += np.sum(np.log(2*math.pi*math.e/self.r_phi))/2
        # E[log P(Psi) - log q(Psi)]
        self.obj += 0
        # E[log P(Z | pi) - log q(Z)]
        idx_pi = (self.Epi != 0) & (self.Epi != 1)
        idx_pz = (self.p_z != 0) & (self.p_z != 1)
        self.obj += self.T * np.sum(self.Epi[idx_pi] * np.log(self.Epi[idx_pi]) + (1-self.Epi[idx_pi]) * np.log(1-self.Epi[idx_pi])) + np.sum(-self.p_z[idx_pz] * np.log(self.p_z[idx_pz]) - (1-self.p_z[idx_pz]) * np.log(1-self.p_z[idx_pz])) 
        # E[log P(pi) - log q(pi)]
        self.obj += 0
        # E[log P(gamma) - log q(gamma)]
        self.obj += 0
        pass
