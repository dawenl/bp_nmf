import sys, math
import numpy as np
import scipy.optimize, scipy.special

class BpNMF:
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
        Given mean and precision of a Gaussian r.v. theta, compute E[exp(theta)], E[exp(2*theta)], and E[exp(-theta)]
        '''
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), np.exp(mu - 1./(2*r)))

    def update(self, time=False, verbose=True):
        print 'Updating DZS...'
        for k in xrange(self.K):
            self.update_phi(k)
            self.update_z_k(k)
            self.update_psi(k, time)
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
                print d['task']
            else:
                print d['warnflag']
        # update expected dictionary
        self.ED[:,k], self.ED2[:,k], _ = self._exp(self.mu_phi[:,k], self.r_phi[:,k])
  
    def update_z(self, k):
        Eres = self.X - np.dot(self.ED, self.ES*self.EZ) + np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        dummy = self.Eg * (-1./2 * np.outer(self.ED2[:,k], self.ES2[k,:]).sum(axis=0) + np.sum(np.outer(self.ED[:,k], self.ES[k,:]) * Eres, axis=0))
        p0 = scipy.special.psi(self.beta_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k])
        p1 = scipy.special.psi(self.alpha_pi[k]) - scipy.special.psi(self.alpha_pi[k] + self.beta_pi[k]) + dummy
        self.p_z[k,:] = 1./(1 + np.exp(p0 - p1))
        self.EZ[k,:] = self.p_z[k,:]

    def update_psi(self, k, time=False):
        if time:
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
        if np.alltrue(self.r_phi[:,k] > 0) == False:
            if d['warnflag'] == 2:
                print d['task']
            else:
                print d['warnflag']
        self.ES[k,:], self.ES2[k,:], self.ESinv[k,:] = self._exp(self.mu_psi[k,:], self.r_psi[k,:])

    def _update_psi_time(self, k):
        pass

    ## WRONG
    #def update_psi(self, t):
    #    def f_eta(psi):
    #        return self.Eg * np.exp(psi) * self.EZ[:,t] * np.sum(self.X[:,t]) * np.sum(self.ED, axis=0)
    #    
    #    def f(psi):
    #        '''
    #        The function to be minimized. Only terms related to Psi_{kt} is kept 
    #        '''
    #        Eeta = f_eta(psi) 
    #        EAeta = self.Eg/2 * (np.exp(2*psi)*self.EZ[:,t]*np.sum(self.ED2, axis=0) + 2 * np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
    #        if t == 0:
    #            # no expectation, just log(P(psi_t))
    #            Ebwd = self.alpha * psi - self.alpha * np.exp(psi) 
    #        else:
    #            Ebwd = self.alpha * psi - self.alpha*self.ESinv[:,t-1]*np.exp(psi)
    #        if t == self.T-1:
    #            Efwd = 0
    #        else:
    #            Efwd = - self.alpha * np.exp(-psi) * self.ESinv[:,t+1] - self.alpha*psi
    #        return -np.sum(Eeta - EAeta + Efwd + Ebwd)

    #    def df(psi):
    #        '''
    #        The first derivative of f(psi)
    #        '''
    #        dEeta = f_eta(psi) 
    #        dEAeta = self.Eg * (np.exp(2*psi)*self.EZ[:,t]*np.sum(self.ED2, axis=0) + np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
    #        if t == 0:
    #            dEbwd = self.alpha - self.alpha * np.exp(psi)
    #        else:
    #            dEbwd = self.alpha - self.alpha * self.ESinv[:,t-1] * np.exp(psi)
    #        if t == self.T-1:
    #            dEfwd = 0
    #        else:
    #            dEfwd = self.alpha * np.exp(-psi) * self.ESinv[:,t+1] - self.alpha
    #        return -(dEeta - dEAeta + dEfwd + dEbwd)

    #    def df2(psi):
    #        '''
    #        The Hessian of f(psi)
    #        '''
    #        dEeta2 = f_eta(psi) 
    #        dEAeta2 = self.Eg * (2*np.exp(2*psi)*self.EZ[:, t]*np.sum(self.ED2, axis=0) + np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
    #        if t == 0:
    #            dEbwd2 = -self.alpha * np.exp(psi)
    #        else:
    #            dEbwd2 = -self.alpha * self.ESinv[:,t-1] * np.exp(psi)
    #        if t == self.T-1:
    #            dEfwd2 = 0
    #        else:
    #            dEfwd2 = -self.alpha * np.exp(-psi) * self.ESinv[:,t+1]
    #        return -(dEeta2 - dEAeta2 + dEfwd2 + dEbwd2)
    #    #idx = (self.EZ[:,t] == 1)

    #    #mu_const = -1./2 * np.log((self.alpha + 1)/self.alpha)
    #    #r_const = 1./(np.log(self.alpha + 1) - np.log(self.alpha))
    #    
    #    #self.mu_psi[:,t], self.r_psi[:,t] = mu_const, r_const
    #    # only update the terms with z = 1, for the remaining just update so that E[S] = 1, E[S^2] = (alpha + 1)/alpha
    #    
    #    dummy = np.dot(self.ED, self.ES[:,t]*self.EZ[:,t])
    #    # F by K matrix with each column as E[X_t^{-k}] = E[D] * (E[S_t] .* E[Z_t]) - E[D_k] * (E[S_{kt}] .* E[Z_{kt}] 
    #    Eres = np.tile(dummy, (self.K, 1)).T - np.dot(self.ED, np.diag(self.ES[:,t] * self.EZ[:,t]))

    #    #psi0 = np.zeros((K, ))
    #    psi0 = self.mu_psi[:,t]
    #    #tmp, _, _ = scipy.optimize.fmin_tnc(f, psi0, fprime=df, disp=5)
    #    tmp, _, _ = scipy.optimize.fmin_l_bfgs_b(f, psi0, fprime=df, disp=5)
    #    self.mu_psi[:,t], self.r_psi[:,t] = tmp, df2(tmp)
    #    
    #    self.ES[:,t], self.ES2[:,t], self.ESinv[:,t] = self._exp(self.mu_psi[:,t], self.r_psi[:,t])

    ## WRONG
    #def update_z(self, t):
    #    # F by K matrix with each column as X_t - E[X_t^{-k}]
    #    Eres = np.tile(self.X[:,t] - np.dot(self.ED, self.ES[:,t]*self.EZ[:,t]), (self.K, 1)).T + np.dot(self.ED, np.diag(self.ES[:,t] * self.EZ[:,t]))
    #    dummy = self.Eg * (-1./2 * self.ES2[:,t] * np.sum(self.ED2, axis=0) + self.ES[:,t] * np.sum(self.ED * Eres, axis=0))
    #    p0 = scipy.special.psi(self.beta_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi)
    #    p1 = scipy.special.psi(self.alpha_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi) + dummy
    #    self.p_z[:,t] = 1./(1 + np.exp(p0 - p1))
    #    #self.EZ[:,t] = np.round(self.p_z[:,t])
    #    self.EZ[:,t] = (p1 >= p0).astype(int)

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
