# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time
import bp_nmf, librosa
from bp_utils import *

# <codecell>

n_fft = 512
hop_length = 256
X = load_data('data/demo.mp3', n_fft, hop_length, sr=22050)
K = 512

# <codecell>

reload(bp_nmf)
threshold = 0.0001
maxiter = 50
objs = []
old_obj = -np.inf
bnmf = bp_nmf.BP_NMF(X, K=K, seed=357)
    
for i in xrange(maxiter):
    start_t = time.time()
    bnmf.update(verbose=True, disp=1)
    t = time.time() - start_t
    objs.append(bnmf.obj)
    improvement = (bnmf.obj - old_obj) / abs(bnmf.obj)
    old_obj = bnmf.obj
    print 'Iteration: {}, good K: {}, time: {:.2f}, obj: {:.2f} (improvement: {:.4f})'.format(i, bnmf.good_k.shape[0], t, bnmf.obj, improvement)
    if improvement < threshold:
        break

# <codecell>

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
plot(objs[1:])
pass

# <codecell>

good_k = bnmf.good_k

Xres = dot(bnmf.ED, bnmf.ES * around(bnmf.EZ))
res = X - Xres

## Original v.s. Reconstruction
figure(1)
gsubplot(args=({'D':logspec(X), 'T':'Original Spectrogram'}, 
            {'D':logspec(Xres), 'T':'Reconstruction '} , 
            {'D':res, 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
idx = flipud(argsort(bnmf.Epi[good_k]))
tmpED = bnmf.ED[:,good_k[idx]].copy()
tmpED /= np.max(tmpED, axis=0, keepdims=True)
tmpES = bnmf.ES[good_k[idx],:].copy()
tmpES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
figure(2)
gsubplot(args=({'D':logspec(tmpED), 'T':'ED'}, 
            {'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, 
            {'D':logspec((around(bnmf.EZ[good_k[idx],:])*tmpES)[:,-1000:]), 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')
title('Expected membership prior Pi')
figure(4)
tmpED = bnmf.ED[:,good_k[idx]].copy()
tmpED /= np.sum(tmpED**2, axis=0)**0.5
gsubplot(args=(dot(tmpED.T, tmpED),), cmap=cm.hot_r)

# <codecell>

tmpEZ = around(bnmf.EZ[good_k[idx],:])
figure(1)
num = len(good_k)
for i in xrange(0,2*num,2):
    subplot(num, 2, i+1)
    plot(10*log10(tmpED[:, i/2]))
    subplot(num, 2, i+2)
    plot((tmpEZ * tmpES)[i/2,:])
tight_layout()

# <codecell>


