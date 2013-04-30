# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle
import bp_vbayes
from plot_utils import *
import librosa

# <codecell>

## Load Pickled data if necessary
filename = 'bnmf_mix5a_R_20T'
with open(filename, 'r') as output:
    bnmf = pickle.load(output)

# <codecell>

#x, sr = librosa.load('../data/mix_var5a_8k.wav', sr=None)
#x, sr = librosa.load('../data/demo.wav')
x, sr = librosa.load('../data/PinkMoon.mp3', sr=22050)
n_fft = 512
hop_length = 512
Xc = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
X = abs(Xc)
K = 512
print X.shape
std_col = np.sqrt(np.var(X, axis=1, keepdims=True)) + 1e-6
X /= std_col

#X /= amax(X)
#X[X < 1e-8] = 1e-8
x_max = amax(X)
X[ X < 1e-8 * x_max] = 1e-8 * x_max

# <codecell>

reload(bp_vbayes)
init_option = 'Rand'
alpha = 2.
bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, RSeed=random.seed(98765), alpha=alpha)

# <codecell>

N = 20
#timed = zeros((N,), dtype='bool')
timed = ones((N,), dtype='bool')
timed[0] = False
#timed[-10:] = True
obj = []
for i in xrange(N):
    start_t = time.time()
    bnmf.update(timed=timed[i])
    t = time.time() - start_t
    print 'Iteration: {}, good K: {}, time: {:.2f}'.format(i,bnmf.good_k.shape[0], t)
    obj.append(bnmf.obj)

# <codecell>

threshold = 0.0001
N = 30
#timed = zeros((N,), dtype='bool')
timed = ones((N,), dtype='bool')
timed[0] = False
#timed[-5:] = True
obj = []
improvement = 1
for i in xrange(N):
    start_t = time.time()
    bnmf.update(timed=timed[i])
    t = time.time() - start_t
    
    if i > 0:
        last_score = score
    score = bnmf.obj
    obj.append(score)
    if i > 0:
        improvement = (score - last_score)/abs(last_score)
    print 'Iteration: {}, good K: {}, time: {:.2f}, improvement: {:.4f}'.format(i, bnmf.good_k.shape[0], t, improvement)
    if improvement < threshold:
        break

# <codecell>

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
print diff(obj)
plot(obj[1:])
pass

# <codecell>

good_k = bnmf.good_k
# Rescaling the data
high_cut = 20 #db
low_cut = -60 #db
threshold_high = 10**(high_cut/20.)
threshold_low = 10**(low_cut/20.)

tmpX = X.copy()
tmpX *= std_col
Xres = dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)) * std_col
res = tmpX - Xres

tmpX /= (np.max(tmpX)/threshold_high)
Xres /= (np.max(Xres)/threshold_high)
tmpX[tmpX < threshold_low] = threshold_low
Xres[Xres < threshold_low] = threshold_low

## Original v.s. Reconstruction
figure(1)
plot_decomp(args=({'D':20*log10(tmpX), 'T':'Original Spectrogram'}, 
            {'D':20*log10(Xres), 'T':'Reconstruction '} , 
            {'D':res, 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
idx = flipud(argsort(bnmf.Epi[good_k]))
tmpED = bnmf.ED.copy()
tmpED /= (np.max(bnmf.ED, axis=0)/threshold_high)
tmpES = bnmf.ES.copy()
tmpES *= (np.max(bnmf.ED, axis=0, keepdims=True).T/threshold_high)
tmpED[tmpED < threshold_low] = threshold_low
figure(2)
plot_decomp(args=({'D':20*log10(tmpED[:,good_k[idx]]), 'T':'ED'}, 
            {'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, 
            {'D':(around(bnmf.EZ)*tmpES)[good_k[idx],:], 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')
title('Expected membership prior Pi')

# <codecell>

bar(arange(std_col.shape[0]), std_col)

# <codecell>

tmpED = bnmf.ED.copy()
tmpED /= np.sum(tmpED**2, axis=0)**0.5
plot_decomp(args=(dot(tmpED[:,good_k[idx]].T, tmpED[:,good_k[idx]]),))

# <codecell>

print amax(bnmf.ESinv), amin(bnmf.ESinv)
print amax(bnmf.ES), amin(bnmf.ES)
print amax(bnmf.ES2), amin(bnmf.ES2)
print amax(bnmf.ED), amin(bnmf.ED)

# <codecell>

## Compare with regular NMF
import pymf
nmf = pymf.NMF(X, num_bases=good_k.shape[0], niter=500)
nmf.initialization()
nmf.factorize()

# <codecell>

figure(1)
nmf.W[nmf.W < 1e-3] = 1e-3
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
        xl.append(librosa.istft(XL, n_fft=n_fft))
    if save:
        sio.savemat('xl.mat', {'xl':np.array(xl)})
    return np.array(xl)
xl_bp = separate(bnmf.ED[:,good_k[idx]], (bnmf.ES*around(bnmf.EZ))[good_k[idx],:])
#xl_nmf = separate(nmf.W, nmf.H)

# <codecell>

sio.savemat('bnmf.mat', {'ED':bnmf.ED[:,good_k[idx]], 'ESZ':around(bnmf.EZ[good_k[idx],:])*bnmf.ES[good_k[idx],:]})

# <codecell>

## only works for 44.1kHz
import scikits.audiolab as audiolab
for i in xrange(good_k.shape[0]):
    audiolab.play(xl_bp[i,:])

# <codecell>

## save as a Picked object
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
save_object(bnmf, 'bnmf_mix5a_R_1N19T_Scale')

# <codecell>


