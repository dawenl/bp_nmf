# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, time, midi, pickle
import scipy.io as sio
import scipy.sparse
import bp_vbayes, librosa, utils

# <codecell>

instruments = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe']
n_inst = len(instruments)
notes = {}
for inst in instruments:
    notes[inst] = utils.midi2notes('../data/midi/{}.mid'.format(inst[0]))
    
notes_count = {}
for inst in instruments:
    notes_count[inst] = len(notes[inst])

# <codecell>

x_inst = None
for i in xrange(n_inst):
    x, sr = librosa.load('../data/{}_var5a22k.wav'.format(instruments[i]))
    num_samples = len(x)
    if x_inst is None:
        x_inst = zeros((n_inst, num_samples))
    x_inst[i,:] = x

# <codecell>

n_fft = 1024
hop_length = 512
n_frames = 1 + int( (num_samples - n_fft) / hop_length)
X_envelope = np.zeros((n_inst, n_frames))

for i in xrange(n_inst):
    X_envelope[i,:] = utils.envelope(x_inst[i,:], n_fft, hop_length)
X_envelope = (X_envelope - np.mean(X_envelope, axis=1, keepdims=True)) / np.sqrt(np.var(X_envelope, axis=1, keepdims=True))

# <codecell>

reload(utils)
pickle_file = 'bnmf_mix_var5a22k_Nscale_20N_F1024_H512_K512_GK46'
bnmf = utils.load_object(pickle_file)
x, sr = librosa.load('../data/mix_var5a22k.wav')
num_samples = len(x)

# <codecell>

good_k = bnmf.good_k
idx = flipud(argsort(bnmf.Epi[good_k]))

tmpES = (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:]
tmpES = (tmpES - np.mean(tmpES, axis=1, keepdims=True)) / np.sqrt(np.var(tmpES, axis=1, keepdims=True))

# <codecell>

for i in xrange(n_inst):
    figure(i)
    subplot(211)
    plot(dot(tmpES, X_envelope[i,:]), '-o')
    subplot(212)
    plot(flipud(sort(dot(tmpES, X_envelope[i,:]))), '-o')

# <codecell>

print X_envelope.shape
print notes_count

# <codecell>

multi2one = False
if multi2one:
    idx_comp = []
    count_comp = []
    for i in xrange(n_inst):
        #tmp = dot(tmpES, X_envelope[i,:])
        #bound = len(good_k)-1-np.argmax(np.diff((np.sort(tmp))))
        #bound = notes_count[instruments[i]]
        bound = len(good_k) / 5
        tmp_idx = flipud(np.argsort(dot(tmpES, X_envelope[i,:])))
        idx_comp.append(tmp_idx[:bound])
        count_comp.extend(tmp_idx[:bound])
        figure(i)
        plot((dot(tmpES, X_envelope[i,:]))[tmp_idx], '-bx')
        plot((dot(tmpES, X_envelope[i,:]))[tmp_idx[:bound]], '-ro')
else:
    idx_comp = np.argmax(dot(tmpES, X_envelope.T), axis=0)
#figure(1)
#hist(np.argmax(dot(tmpES, X_envelope.T), axis=0), bins=len(good_k))
#figure(2)
#hist(count_comp, bins=len(good_k))
pass

print idx_comp

# <codecell>

x_sep = []
ratio = 2 * hop_length / n_fft;
for i in xrange(n_inst):
    mask = utils.wiener_mask(bnmf.ED[:,good_k[idx]], (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:], idx=idx_comp[i])
    if ratio == 2:
        window = zeros((2*n_fft,))
        window[:n_fft] = 1
        mask_interp = utils.interp_mask(mask)
    else:
        window = None
        mask_interp = mask
    X_ola = librosa.stft(x, n_fft=ratio*n_fft, window=window, hop_length=hop_length)
    x_sep.append(librosa.istft(X_ola * mask_interp[:,:X_ola.shape[1]], n_fft=ratio*n_fft, hop_length=hop_length, window=window))
x_sep = np.array(x_sep)

# <codecell>

x_res = np.zeros((n_inst, num_samples))
x_res[:, :x_sep.shape[1]] = x_sep

# <codecell>

snr = []
for i in xrange(n_inst):
    noise = x_res[i,:] - x_inst[i,:]
    snr.append(10*log10(mean(x_inst[i,:]**2)/mean(noise**2)))
print 'Average SNR: {} +- {}'.format(mean(snr), sqrt(var(snr)))
hist(snr, bins=20)
pass

# <codecell>

sio.savemat('inst.mat', {'x_res':x_res})

