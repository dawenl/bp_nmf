# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import bp_vbayes
from plot_utils import *

# <codecell>

# generate data
random.seed(13579)
K = 72
L = 9
F = 36
T = 300
D = exp(randn(F, L))
#S = empty((L, T))
#S[:,0] = random.gamma(2, scale=1./2, size=(L,))
#for t in xrange(1, T):
#    S[:, t] = random.gamma(2, scale=S[:,t-1]/2., size=(L,)) 
S = random.gamma(2, scale=1./2, size=(L,T))
Z = ones((L,T))
#Z = random.binomial(1, 0.8, size=(L,T))
X = dot(D, S*Z)  
plot_decomp(args=(D, (S*Z), X))

# <codecell>

import librosa
x, sr = librosa.load('../data/dawen.wav')
Xc = librosa.stft(x, n_fft=512)
X = abs(Xc)
K = 512

# <codecell>

reload(bp_vbayes)
bnmf = bp_vbayes.Bp_NMF(X, K=K, RSeed=random.seed(357))

# <codecell>

N = 10
timed = zeros((N,), dtype='bool')
#timed = ones((N,), dtype='bool')
timed[-6:] = True
good_k = np.arange(bnmf.K)
obj = []
for i in xrange(N):
    print 'Iteration: {}, good K: {}'.format(i, good_k.shape[0])
    for k in good_k:
        bnmf.update_phi(k)
        bnmf.update_z(k)
        bnmf.update_psi(k, timed=timed[i])
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

cutoff= 60 #db
threshold = 10**(-cutoff/20)
## Original v.s. Reconstruction
X[X < threshold] = threshold
Xres = dot(bnmf.ED, bnmf.ES * around(bnmf.EZ))
Xres[Xres < threshold] = threshold
figure(1)
plot_decomp(args=({'D':20*log10(X), 'T':'Original Spectrogram'}, 
            {'D':20*log10(Xres), 'T':'Reconstruction '} , 
            {'D':X-dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)), 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
idx = flipud(argsort(bnmf.Epi[good_k]))
tmpED = bnmf.ED.copy()
tmpED[tmpED < threshold] = threshold
figure(2)
plot_decomp(args=({'D':20*log10(tmpED[:,good_k[idx]]), 'T':'ED'}, 
            {'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, 
            {'D':(around(bnmf.EZ)*bnmf.ES)[good_k[idx],:], 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')

# <codecell>

idx = flipud(argsort(bnmf.Epi[good_k]))
figure(1)
plot_decomp(args=({'D':bnmf.ED[:,good_k[idx]], 'T':'ED'}, {'D':D[:,:L], 'T':'D'}))
figure(2)
plot_decomp(args=({'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, {'D':Z[:good_k.shape[0],:], 'T':'Z'}))
figure(3)
plot_decomp(args=({'D':(around(bnmf.EZ)*bnmf.ES)[good_k[idx],:], 'T':'ES*EZ'}, {'D':(S*Z)[:good_k.shape[0],:], 'T':'S*Z'}))

# <codecell>

good_k.shape

# <codecell>

print amax(bnmf.ESinv), amin(bnmf.ESinv)
print amax(bnmf.ES), amin(bnmf.ES)
print amax(bnmf.ES2), amin(bnmf.ES2)
print amax(bnmf.ED), amin(bnmf.ED)

# <codecell>

import pymf
nmf = pymf.NMF(X, num_bases=10, niter=100)
nmf.initialization()
nmf.factorize()

# <codecell>

figure(1)
plot_decomp(args=(20*log10(nmf.W), 20*log10(bnmf.ED[:,good_k])), cmap=cm.hot_r)
figure(2)
plot_decomp(args=(nmf.H, bnmf.ES[good_k,:]*around(bnmf.EZ[good_k,:])), cmap=cm.hot_r)

# <codecell>

import scipy.io as sio
def separate(W, H, save=True):
    xl = [] 
    den = np.dot(W, H)
    den[den==0] += 1e-6
    L = W.shape[1]
    for l in xrange(L): 
        XL = Xc * np.outer(W[:,l], H[l,:])/den
        xl.append(librosa.istft(XL, n_fft=512))
    if save:
        sio.savemat('xl.mat', {'xl':np.array(xl)})
    return np.array(xl)
xl_bp = separate(bnmf.ED[:,good_k], bnmf.ES[good_k,:]*around(bnmf.EZ[good_k,:]))
#xl_nmf = separate(nmf.W, nmf.H)

# <codecell>

import scikits.audiolab as audiolab
for i in xrange(good_k.shape[0]):
    audiolab.play(xl_bp[i,:])

# <codecell>


