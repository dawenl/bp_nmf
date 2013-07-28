# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

""" Toy example for BP-NMF 
"""
import time

import bp_nmf
from bp_utils import *
import librosa

# <codecell>

n_fft = 512
hop_length = 256
X = load_data('data/demo.mp3', n_fft, hop_length, sr=22050)
K = 512

# <codecell>

threshold = 0.0001
maxiter = 100
objs = []
old_obj = -np.inf
bnmf = bp_nmf.LVI_BP_NMF(X, K=K, seed=357)

## Each dot represent 20 updated components, for monitoring the progress
    
for i in xrange(maxiter):
    start_t = time.time()
    bnmf.update(verbose=True, disp=1)
    t = time.time() - start_t
    objs.append(bnmf.obj)
    improvement = (bnmf.obj - old_obj) / abs(old_obj)
    old_obj = bnmf.obj
    print 'Iteration: {}, good K: {}, time: {:.2f}, obj: {:.2f} (improvement: {:.4f})'.format(i, bnmf.good_k.size, t, bnmf.obj, improvement)
    if improvement < threshold:
        break

# <codecell>

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
plot(objs[1:])
pass

# <codecell>

good_k = bnmf.good_k

idx = flipud(argsort(bnmf.Epi[good_k]))
# normalize the dictionary so that each component has maximum 1
ED = bnmf.ED[:,good_k[idx]]
ED /= np.max(ED, axis=0, keepdims=True)
ES = bnmf.ES[good_k[idx],:]
ES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
EZ = around(bnmf.EZ[good_k[idx],:])

Xres = dot(ED, ES * EZ)
res = X - Xres

## Original v.s. Reconstruction
figure(1)
gsubplot(args=({'D':logspec(X), 'T':'Original Spectrogram'}, 
               {'D':logspec(Xres), 'T':'Reconstruction '} , 
               {'D':res, 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
figure(2)
gsubplot(args=({'D':logspec(ED), 'T':'ED'}, 
               {'D':EZ, 'T':'EZ'}, 
               {'D':logspec(EZ*ES), 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')
title('Expected membership prior Pi')
pass

# <codecell>

figure(1)
num = len(good_k)
for i in xrange(0, 2*num, 2):
    subplot(num, 2, i+1)
    plot(10*log10(ED[:, i/2]))
    subplot(num, 2, i+2)
    plot((EZ * ES)[i/2,:])
tight_layout()

# <codecell>


