# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
Instrument-level BSS eval
"""
import sys, time, midi, pickle
import numpy as np
import scipy.io as sio
import scipy.sparse
import bp_nmf, librosa
from bp_utils import *

# <codecell>

# obtain the note-on timing for each instrument from midi files
instruments = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe']
n_inst = len(instruments)
notes = {}
for inst in instruments:
    notes[inst] = midi2notes('data/midi/{}.mid'.format(inst[0]))
    
notes_count = {}
for inst in instruments:
    notes_count[inst] = len(notes[inst])

# <codecell>

# extract the notes from signals based on the timing from midi files
x_inst = None
for i in xrange(n_inst):
    x, sr = librosa.load('data/{}_var5a22k.wav'.format(instruments[i]))
    num_samples = len(x)
    if x_inst is None:
        x_inst = np.zeros((n_inst, num_samples))
    x_inst[i,:] = x

# <codecell>

# compute the envelope for extracted instrument-level signals
n_fft = 1024
hop_length = 512
n_frames = 1 + int( (num_samples - n_fft) / hop_length)
X_envelope = np.zeros((n_inst, n_frames))

for i in xrange(n_inst):
    X_envelope[i,:] = envelope(x_inst[i,:], n_fft, hop_length)
X_envelope = (X_envelope - np.mean(X_envelope, axis=1, keepdims=True)) / np.sqrt(np.var(X_envelope, axis=1, keepdims=True))
print X_envelope.shape

# <codecell>

## Load the pre-saved models if any ##
pickle_file = 'bpnmf_mix_F1024_H512_K512'
bp_nmf = load_object(pickle_file)
x, sr = librosa.load('data/mix_var5a22k.wav')
num_samples = len(x)

# <codecell>

# compute the correlation between activations and instrument-level signal envelope
good_k = bp_nmf.good_k
idx = np.flipud(np.argsort(bp_nmf.Epi[good_k]))

tmpES = (np.around(bp_nmf.EZ) * bp_nmf.ES)[good_k[idx],:]
tmpES = (tmpES - np.mean(tmpES, axis=1, keepdims=True)) / np.sqrt(np.var(tmpES, axis=1, keepdims=True))

for i in xrange(n_inst):
    figure(i)
    subplot(211)
    plot(np.dot(tmpES, X_envelope[i,:]), '-o')
    title('Correlation: E[S]*E[Z] v.s. {} envelope'.format(instruments[i]))
    xlabel('components index')
    ylabel('correlation')
    subplot(212)
    plot(np.flipud(np.sort(np.dot(tmpES, X_envelope[i,:]))), '-o')
    title('Sorted correlation: E[S]*E[Z] v.s. {} envelope'.format(instruments[i]))
    ylabel('correlation')
    tight_layout()

# <codecell>

# if multi2one true, each instrument will be corresponding to size(good_k)/5 components according to the correlation
# otherwise, each instrument will be corresponding to a single component according to the correlation

multi2one = False
if multi2one:
    idx_comp = []
    count_comp = []
    for i in xrange(n_inst):
        bound = len(good_k) / 5
        tmp_idx = flipud(np.argsort(dot(tmpES, X_envelope[i,:])))
        idx_comp.append(tmp_idx[:bound])
        count_comp.extend(tmp_idx[:bound])
        figure(i)
        plot((dot(tmpES, X_envelope[i,:]))[tmp_idx], '-bx')
        plot((dot(tmpES, X_envelope[i,:]))[tmp_idx[:bound]], '-ro')
else:
    idx_comp = np.argmax(dot(tmpES, X_envelope.T), axis=0)

# <codecell>

# Now separate out the instrument-level signal from mixed signal via Wiener filter
# If n_fft == hop_length, for better reconstruction quality, we will do interpolation on the mask
x_sep = []
ratio = 2 * hop_length / n_fft;
for i in xrange(n_inst):
    mask = wiener_mask(bp_nmf.ED[:,good_k[idx]], (around(bp_nmf.EZ) * bp_nmf.ES)[good_k[idx],:], idx=idx_comp[i])
    if ratio == 2:
        # interpolation the mask for better reconstruction quality
        window = zeros((2*n_fft,))
        window[:n_fft] = 1
        mask_interp = interp_mask(mask)
    else:
        window = None
        mask_interp = mask
    X_ola = librosa.stft(x, n_fft=ratio*n_fft, window=window, hop_length=hop_length)
    x_sep.append(librosa.istft(X_ola * mask_interp[:,:X_ola.shape[1]], n_fft=ratio*n_fft, hop_length=hop_length, window=window))
x_sep = np.array(x_sep)
x_res = np.zeros((n_inst, num_samples))
x_res[:, :x_sep.shape[1]] = x_sep

# <codecell>

# this can be used for bss-eval in MATLAB
sio.savemat('inst.mat', {'x_sep':x_res})

