# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle, functools
import bp_vbayes
import utils
import librosa
import scipy.io as sio

# <codecell>

## Load Pickled data if necessary
#pickle_file = 'exp/transcription/bnmf_mix_Nscale_20N_GK63'
pickle_file = 'bnmf_mix_Nscale_20N_GK9'
bnmf = utils.load_object(pickle_file)
X = bnmf.X
std_col = 1.

# <codecell>

## Data pre-processing parameters
ID = 'demo'
filename = '{}.mp3'.format(ID)
n_fft = 512
hop_length = 256
reweight = False
X, std_col = utils.get_data(filename, n_fft, hop_length, sr=22050, reweight=reweight)

## Model training parameters
init_option = 'Rand'
alpha = 2.
K = 512
N = 10

# <codecell>

objs = empty((N,))
bnmf = utils.train(X, K, init_option, alpha, N, objs=objs)

print 'sigma_error = {}'.format(sqrt(1./bnmf.Eg))
plot(objs[1:])
pass

# <codecell>

good_k = bnmf.good_k

Xres = dot(bnmf.ED, bnmf.ES * around(bnmf.EZ)) * std_col
res = X*std_col - Xres

## Original v.s. Reconstruction
figure(1)
utils.gsubplot(args=({'D':utils.logspec(X*std_col), 'T':'Original Spectrogram'}, 
            {'D':utils.logspec(Xres), 'T':'Reconstruction '} , 
            {'D':res, 'T':'Reconstruction Error'}), cmap=cm.hot_r)
## Plot decomposition
idx = flipud(argsort(bnmf.Epi[good_k]))
tmpED = bnmf.ED[:,good_k[idx]].copy()
tmpED /= np.max(tmpED, axis=0, keepdims=True)
tmpES = bnmf.ES[good_k[idx],:].copy()
tmpES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
figure(2)
utils.gsubplot(args=({'D':utils.logspec(tmpED), 'T':'ED'}, 
            {'D':around(bnmf.EZ[good_k[idx],:]), 'T':'EZ'}, 
            {'D':utils.logspec((around(bnmf.EZ[good_k[idx],:])*tmpES)[:,-1000:]), 'T':'ES*EZ'}), cmap=cm.hot_r)
figure(3)
plot(flipud(sort(bnmf.Epi[good_k])), '-o')
title('Expected membership prior Pi')
figure(4)
tmpED = bnmf.ED[:,good_k[idx]].copy()
tmpED /= np.sum(tmpED**2, axis=0)**0.5
utils.gsubplot(args=(dot(tmpED.T, tmpED),), cmap=cm.hot_r)

# <codecell>

tmpEZ = around(bnmf.EZ[good_k[idx],:])
figure(1)
num = len(good_k)
#num = 7
for i in xrange(0,2*num,2):
    subplot(num, 2, i+1)
    plot(10*log10(tmpED[:, i/2]))
    subplot(num, 2, i+2)
    plot((tmpEZ * tmpES)[i/2,:])
tight_layout()

# <codecell>

## Save!!
reload(utils)
name = utils.gen_save_name(ID, reweight, n_fft, hop_length, K, good_k=bnmf.good_k.shape[0])
utils.save_object(bnmf, name)

# <headingcell level=1>

# Compare with regular NMF

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

x, _ = librosa.load(filenam)
Xc = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

xl_bp = utils.separate(Xc, bnmf.ED[:,good_k[idx]], (bnmf.ES*around(bnmf.EZ))[good_k[idx],:], n_fft, save=True)
#xl_nmf = utils.separate(nmf.W, nmf.H, save=False)

# <codecell>

## only works for 44.1kHz
import scikits.audiolab as audiolab
for i in xrange(good_k.shape[0]):
    audiolab.play(xl_bp[i,:])
    time.sleep(1)
    #audiolab.play(xl_nmf[i,:])

