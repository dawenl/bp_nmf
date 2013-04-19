# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import bp_vbayes
from plot_utils import *

# <codecell>

# generate data
random.seed(3579)
K = 128
L = 5
F = 60
T = 40
D = exp(randn(F, K))
S = empty((K, T))
S[:,0] = random.gamma(2, scale=1./2, size=(K,))
for t in xrange(1, T):
    S[:, t] = random.gamma(2, scale=S[:,t-1]/2., size=(K,)) 
Z = zeros((K, T))
Z[:5,:] = 1
X = dot(D, S*Z)  
plot_decomp(D, (S*Z)[:L,:], X)

# <codecell>

reload(bp_vbayes)
bnmf = bp_vbayes.BpNMF(X, K=K, RSeed=random.seed(123))

# <codecell>

N = 10
good_k = np.arange(bnmf.K)
obj = []
for i in xrange(N):
    for k in good_k:
        bnmf.update_phi(k)
        bnmf.update_z(k)
        bnmf.update_psi(k)
    bnmf.update_pi()
    bnmf.update_r()
    good_k = np.delete(good_k, np.where(np.sum(bnmf.EZ[good_k,:], axis=1) < 1e-16))
    bnmf._lower_bound()
    obj.append(bnmf.obj)

# <codecell>

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
print diff(obj)
plot(obj)
pass

# <codecell>

plot_decomp({'D':X, 'T':'Original Spectrogram'}, {'D':dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)), 'T':'Reconstruction '} , {'D':X-dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)), 'T':'Reconstruction Error'})

# <codecell>

plot(flipud(sort(bnmf.Epi))[:5*L], '-o')

# <codecell>

idx = flipud(argsort(bnmf.Epi))
plot_decomp(bnmf.ED[:,idx[:3*L]], around(bnmf.EZ[idx[:3*L],:]), (around(bnmf.EZ)*bnmf.ES)[idx[:3*L],:])

# <codecell>

good_k.shape

# <codecell>

print sum(bnmf.ED), sum(D)

# <codecell>

plot_decomp(np.dot(bnmf.ED[:,good_k].T, bnmf.ED[:,good_k]))

# <codecell>


