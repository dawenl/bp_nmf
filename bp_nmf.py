# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle, functools
import bp_vbayes
from plot_utils import *
import librosa
import scipy.io as sio

# <codecell>

'''
## Load Pickled data if necessary
filename = ''
with open(filename, 'r') as output:
    bnmf = pickle.load(output)
'''

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

def logspec(X, amin=1e-10, dbdown=80):
    logX = 20*log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)
    
## save as a Picked object
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# <codecell>

def train(X, K, init_option, alpha, N, timed, objs=None):
    bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, alpha=alpha)
    for n in xrange(N):
        start_t = time.time()
        ind = bnmf.update(timed=timed[n])
        if ind == -1:
            if n == 0 or n == 1:
                # the initialization can be bad and the first iteration will suck, so restart
                print '***Bad initial values, restart***'
                return train(X, K, init_option, alpha, N, timed)
            else:
                # this should rarely happen
                print 'oops'
                sys.exit(-1)
        t = time.time() - start_t
        if objs is not None:
            objs[n] = bnmf.obj
        print 'Dictionary Learning: Iteration: {}, good K: {}, time: {:.2f}'.format(n, bnmf.good_k.shape[0], t)
    return bnmf

# <codecell>

def get_data(filename, n_fft, hop_length, reweight=True):
    x, _ = librosa.load(filename)
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    print X.shape
    specshow(logspec(X))
    if reweight:
        std_col = np.maximum(np.sqrt(np.var(X, axis=1, keepdims=True)), 1e-10)
    else:
        std_col = 1.
    X /= std_col
    # cut off values 80db below maximum for numerical consideration
    x_max = X.max()
    X[ X < 1e-8 * x_max] = 1e-8 * x_max
    return (X, std_col)

# <codecell>

#reload(bp_vbayes)
filename = '../data/mix_var5a_22k.wav'
n_fft = 512
hop_length = 512
reweight = False
X, std_col = get_data(filename, n_fft, hop_length, reweight=reweight)

# <codecell>

init_option = 'Rand'
alpha = 2.
K = 512
N = 20
#timed = zeros((N,), dtype='bool')
timed = ones((N,), dtype='bool')
timed[0] = False
objs = empty((N,))

bnmf = train(X, K, init_option, alpha, N, timed, objs)

# <codecell>

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
print diff(obj)
plot(obj)
pass

# <codecell>

good_k = bnmf.good_k

Xres = dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)) * std_col
res = X*std_col - Xres

## Original v.s. Reconstruction
figure(1)
plot_decomp(args=({'D':logspec(X*std_col), 'T':'Original Spectrogram'}, 
            {'D':logspec(Xres), 'T':'Reconstruction '} , 
            {'D':res, 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
idx = flipud(argsort(bnmf.Epi[good_k]))
tmpES = bnmf.ES[good_k[idx],:].copy()
tmpES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
figure(2)
plot_decomp(args=({'D':logspec(bnmf.ED[:,good_k[idx]]), 'T':'ED'}, 
            {'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, 
            {'D':around(bnmf.EZ[good_k[idx],:])*tmpES, 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')
title('Expected membership prior Pi')
pass

# <codecell>

tmpED = bnmf.ED[:,good_k[idx]].copy()
tmpED /= np.sum(tmpED**2, axis=0)**0.5
plot_decomp(args=(dot(tmpED.T, tmpED),), cmap=cm.hot_r)

# <codecell>

save_object(bnmf, 'bnmf_mix5a22k_GK_{}_1N19T_Nscale'.format(bnmf.good_k.shape[0])

# <codecell>

## Compare with regular NMF
import pymf
nmf = pymf.NMF(X, num_bases=good_k.shape[0], niter=500)
nmf.initialization()
nmf.factorize()

figure(1)
nmf.W[nmf.W < 1e-3] = 1e-3
plot_decomp(args=(20*log10(nmf.W), 20*log10(bnmf.ED[:,good_k])), cmap=cm.hot_r)
figure(2)
plot_decomp(args=(nmf.H, bnmf.ES[good_k,:]*around(bnmf.EZ[good_k,:])), cmap=cm.hot_r)

# <codecell>

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
xl_bp = separate(bnmf.ED[:,good_k[idx]], (bnmf.ES*around(bnmf.EZ))[good_k[idx],:], save=False)
xl_nmf = separate(nmf.W, nmf.H, save=False)

# <codecell>

## only works for 44.1kHz
import scikits.audiolab as audiolab
for i in xrange(good_k.shape[0]):
    audiolab.play(xl_bp[i,:])
    time.sleep(1)
    #audiolab.play(xl_nmf[i,:])

# <codecell>


