import sys
import numpy as np
import pymf
import scipy.optimize, scipy.special

DEBUG = True

class BpNMF:
    def __init__(self, X, K=512, **kwargs):
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
            self.alpha_g, self.beta_g = 1e-6, 1e-6

            # init the expectations
            self.ED, self.ED2, _ = self._exp(self.mu_phi, self.r_phi)
            self.ES, self.ES2, self.ESinv = self._exp(self.mu_psi, self.r_psi)
            self.EZ = np.round(self.p_z)
            self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)
            self.Eg = self.alpha_g / self.beta_g

        if init_option == 'NMF':
            print 'Init with NMF...'
            nmf = pymf.NMF(self.X, num_bases=num_bases, show_progress=DEBUG)
            #nmf.initialization()
            nmf.factorize()

    def _exp(self, mu, r):
        '''
        Given mean and precision of a Gaussian r.v. theta, compute E[exp(theta)], E[exp(2*theta)], and E[exp(-theta)]
        '''
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), np.exp(mu - 1./(2*r)))

    def update(self, verbose=True):
        print 'Updating D(Phi)...'
        for k in xrange(self.K):
            self.update_phi(k)
            if verbose:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating S(Psi) and Z...' 
        for t in xrange(self.T):
            self.update_z(t)
            self.update_psi(t)
            if verbose:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        print 'Updating pi and gamma...'
        self.update_pi()
        self.update_r()
        self._lower_bound()
        
    def update_phi(self, k):
        Eres = np.dot(self.ED, self.ES*self.EZ) - np.outer(self.ED[:,k], self.ES[k,:]*self.EZ[k,:])
        def f_stub(phi):
            dummy = np.outer(np.exp(phi), self.ES[k,:]*self.EZ[k,:])
            out_prod = np.outer(np.exp(2*phi), self.ES2[k,:]*self.EZ[k,:])
            return (self.Eg * np.sum(self.X * dummy, axis=1), dummy, out_prod)

        def f(phi):
            '''
            The function to be minimized. Only terms related to Phi_{fk} is kept 
            '''
            Eeta, dummy, out_prod = f_stub(phi) 
            EAeta = self.Eg/2 * np.sum(out_prod + 2* dummy * Eres, axis=1)
            return -np.sum(Eeta - EAeta - phi**2/2.) 

        def df(phi):
            '''
            The first derivative of f(phi)
            '''
            dEeta, dummy, out_prod = f_stub(phi) 
            dEAeta = self.Eg * np.sum(out_prod + dummy * Eres, axis=1)
            return -(dEeta - dEAeta - phi)

        def df2(phi):
            '''
            The Hessian of f(phi)
            '''
            dEeta2, dummy, out_prod = f_stub(phi)
            dEAeta2 = self.Eg * np.sum(2*out_prod + dummy * Eres, axis=1)
            return -(dEeta2 - dEAeta2 - 1)
    
        phi0 = np.zeros((self.F,))
        #mu_hat = scipy.optimize.fmin_ncg(f, phi0, df, fhess=df2, maxiter=500, disp=DEBUG)
        mu_hat, _, _ = scipy.optimize.fmin_tnc(f, phi0, fprime=df, disp=5)
        self.mu_phi[:,k], self.r_phi[:,k] = mu_hat, df2(mu_hat)
        # update expected dictionary
        self.ED[:,k], self.ED2[:,k], _ = self._exp(self.mu_phi[:,k], self.r_phi[:,k])
        return df(mu_hat)

    def update_psi(self, t):
        dummy = np.dot(self.ED, self.ES[:,t]*self.EZ[:,t])
        # F by K matrix with each column as E[X_t^{-k}] = E[D] * (E[S_t] .* E[Z_t]) - E[D_k] * (E[S_{kt}] .* E[Z_{kt}] 
        Eres = np.tile(dummy, (self.K, 1)).T - np.dot(self.ED, np.diag(self.ES[:,t] * self.EZ[:,t]))
            
        def f_eta(psi):
            '''
            Compute natural parameters of P(psi), and since only exp(psi) appears in natural parameters, this can also be used to compute the first/second derivative
            '''
            return self.Eg * np.exp(psi) * self.EZ[:,t] * np.sum(self.X[:,t]) * np.sum(self.ED, axis=0)
        
        def f(psi):
            '''
            The function to be minimized. Only terms related to Psi_{kt} is kept 
            '''
            Eeta = f_eta(psi) 
            EAeta = self.Eg/2 * (np.exp(2*psi)*self.EZ[:,t]*np.sum(self.ED2, axis=0) + 2 * np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
            if t == 0:
                # no expectation, just log(P(psi_t))
                Ebwd = self.alpha * psi - self.alpha * np.exp(psi) 
            else:
                Ebwd = self.alpha * psi - self.alpha*self.ESinv[:,t-1]*np.exp(psi)
            if t == self.T-1:
                Efwd = 0
            else:
                Efwd = - self.alpha * np.exp(-psi) * self.ESinv[:,t+1] - self.alpha*psi
            return -np.sum(Eeta - EAeta + Efwd + Ebwd)

        def df(psi):
            '''
            The first derivative of f(psi)
            '''
            dEeta = f_eta(psi) 
            dEAeta = self.Eg * (np.exp(2*psi)*self.EZ[:,t]*np.sum(self.ED2, axis=0) + np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
            if t == 0:
                dEbwd = self.alpha - self.alpha * np.exp(psi)
            else:
                dEbwd = self.alpha - self.alpha * self.ESinv[:,t-1] * np.exp(psi)
            if t == self.T-1:
                dEfwd = 0
            else:
                dEfwd = self.alpha * np.exp(-psi) * self.ESinv[:,t+1] - self.alpha
            return -(dEeta - dEAeta + dEfwd + dEbwd)

        def df2(psi):
            '''
            The Hessian matrix of f(psi)
            '''
            dEeta2 = f_eta(psi) 
            dEAeta2 = self.Eg * (2*np.exp(2*psi)*self.EZ[:,t]*np.sum(self.ED2, axis=0) + np.exp(psi)*self.EZ[:,t]*np.sum(self.ED * Eres, axis=0))
            if t == 0:
                dEbwd2 = -self.alpha * np.exp(psi)
            else:
                dEbwd2 = -self.alpha * self.ESinv[:,t-1] * np.exp(psi)
            if t == self.T-1:
                dEfwd2 = 0
            else:
                dEfwd2 = -self.alpha * np.exp(-psi) * self.ESinv[:,t+1]
            return -(dEeta2 - dEAeta2 + dEfwd2 + dEbwd2)

        psi0 = np.zeros((self.K,)) 
        #mu_hat = scipy.optimize.fmin_ncg(f, psi0, df, fhess=df2, maxiter=500, disp=DEBUG)
        mu_hat, _, _ = scipy.optimize.fmin_tnc(f, psi0, fprime=df, disp=5)
        self.mu_psi[:,t], self.r_psi[:,t] = mu_hat, df2(mu_hat)
        self.ES[:,t], self.ES2[:,t], self.ESinv[:,t] = self._exp(self.mu_psi[:,t], self.r_psi[:,t])

    def update_z(self, t):
        # F by K matrix with each column as X_t - E[X_t^{-k}]
        Eres = np.tile(self.X[:,t] - np.dot(self.ED, self.ES[:,t]*self.EZ[:,t]), (self.K, 1)).T + np.dot(self.ED, np.diag(self.ES[:,t] * self.EZ[:,t]))

        dummy = self.Eg * (-1./2 * self.ES2[:,t] * np.sum(self.ED2, axis=0) + self.ES[:,t] * np.sum(self.ED * Eres, axis=0))
        p0 = scipy.special.psi(self.beta_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi)
        p1 = scipy.special.psi(self.alpha_pi) - scipy.special.psi(self.alpha_pi + self.beta_pi) + dummy
        self.p_z[:,t] = np.exp(p1)/(np.exp(p0) + np.exp(p1))
        #self.EZ[:,t] = np.round(self.p_z[:,t])
        self.EZ[:,t] = (p1 >= p0).astype(int)

    def update_pi(self):
        self.alpha_pi = self.a0/self.K + np.sum(self.EZ, axis=1)
        self.beta_pi = self.b0 * (self.K-1) / self.K + self.T - np.sum(self.EZ, axis=1)
        self.Epi = self.alpha_pi / (self.alpha_pi + self.beta_pi)

    def update_r(self):
        self.alpha_g = self.c0 + 1./2 * self.F * self.T
        self.beta_g = self.d0 + 1./2 * np.sum((self.X - np.dot(self.ED, self.ES * self.EZ))**2)
        self.Eg = self.alpha_g / self.beta_g

    def _lower_bound(self):
        pass
            
