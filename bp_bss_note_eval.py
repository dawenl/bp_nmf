# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, time, functools, midi, pickle
import scipy.io as sio
import scipy.sparse
import bp_vbayes, librosa, utils

# <codecell>

instruments = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe']
notes = {}
for inst in instruments:
    notes[inst] = utils.midi2notes('../data/midi/{}.mid'.format(inst[0]))

# <codecell>

n_notes = 0
for (inst, note) in notes.items():
    n_notes += len(note)
    
base_idx = 0
data = []
row = []
col = []

for inst in instruments:
    x, sr = librosa.load('../data/{}_var5a22k.wav'.format(inst))
    num_samples = len(x)               
    notes_inst = notes[inst]
    for i, key in enumerate(notes_inst.keys()):
        data_row = []
        n_event = len(notes_inst[key])/2
        for j in xrange(n_event):
            dur = (np.arange(round(sr * notes_inst[key][2*j]), round(sr * notes_inst[key][2*j+1])+1)).astype(int)
            data_row.extend(x[dur])
            col.extend(dur)
        data_row = data_row / np.sqrt(np.sum((data_row - sum(data_row) / num_samples)**2)/num_samples)
        row.extend(len(data_row) * [base_idx + i])
        data.extend(data_row)
    base_idx += len(notes_inst)
x_notes = scipy.sparse.csr_matrix(scipy.sparse.coo_matrix((data, (row, col)), shape=(n_notes, num_samples)))

# <codecell>

for i in xrange(n_notes): 
    tt = x_notes.getrow(i)
    print np.sqrt(np.var(TTTT[i,:])), np.sqrt(np.var(tt.todense())), allclose(tt.todense(), TTTT[i,:])

# <codecell>

n_fft = 512
hop_length = 512
n_frames = 1 + int( (num_samples - n_fft) / hop_length)
X_envelope = np.zeros((n_notes, n_frames))

for i in xrange(n_notes):
    X_envelope[i,:] = utils.envelope((x_notes.getrow(i).T).todense(), n_fft, hop_length)

# <codecell>

pickle_file = 'exp/transcription/bnmf_mix_Nscale_20N_GK63'
bnmf = utils.load_object(pickle_file)

# <codecell>

good_k = bnmf.good_k
idx = flipud(argsort(bnmf.Epi[good_k]))

tmpES = (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:]
#tmpES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
tmpES *= np.sum(bnmf.ED[:,good_k[idx]]**2, axis=0, keepdims=True).T**0.5

# <codecell>

for i in xrange(n_notes):
    figure(i)
    #subplot(211)
    #plot(dot(tmpES, X_envelope[i,:]), '-o')
    #subplot(212)
    plot(flipud(sort(dot(tmpES, X_envelope[i,:]))), '-o')

# <codecell>

base_idx = 0
for inst in instruments:
    notes_inst = notes[inst]
    print '{}:'.format(inst)
    for i, key in enumerate(notes_inst.keys()):
        print 'Idx: {}, Pitch: {}, N_notes: {}'.format(argmax(dot(tmpES, X_envelope[base_idx + i,:])), key, len(notes_inst[key])/2)
    base_idx += len(notes_inst)

# <codecell>

idx_comp = np.argmax(dot(tmpES, X_envelope.T), axis=0)
hist(idx_comp, bins=len(good_k))
x, sr = librosa.load('../data/mix_var5a22k.wav')
Xc = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

# <codecell>

MATLAB = True

if MATLAB:
    ## load from MATLAB
    d = sio.loadmat('x_sep1.mat')
    x_sep = d['x_sep']
    del d
else:
    x_sep = utils.separate(Xc, bnmf.ED[:,good_k[idx]], (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:], n_fft, hop_length, idx_comp=idx_comp, save=False)

# <codecell>

data = []
row = []
col = []
base_idx = 0
for inst in instruments:
    notes_inst = notes[inst]
    for i, key in enumerate(notes_inst.keys()):
        data_row = []
        n_event = len(notes_inst[key])/2
        for j in xrange(n_event):
            dur = (np.arange(round(sr * notes_inst[key][2*j]), round(sr * notes_inst[key][2*j+1])+1)).astype(int)
            data_row.extend(x_sep[base_idx + i, dur])
            col.extend(dur)
        data_row = data_row / np.sqrt(np.sum((data_row - sum(data_row) / num_samples)**2)/num_samples)
        row.extend(len(data_row) * [base_idx + i])
        data.extend(data_row)
    base_idx += len(notes_inst)
x_res = scipy.sparse.csr_matrix(scipy.sparse.coo_matrix((data, (row,col)), shape=x_notes.shape))
del x_sep

# <codecell>

x_res = np.zeros((n_notes, num_samples))
x_res[:, :x_sep.shape[1]] = x_sep
std_res = np.sqrt(np.var(x_res, axis=1, keepdims=True))
x_res = x_res / std_res
del x_sep

# <codecell>

np.var(x_res[0,:])

# <codecell>

snr = []
for i in xrange(n_notes):
    noise = x_res.getrow(i).toarray() - x_notes.getrow(i).toarray()
    #noise = x_res[i,:] - x_notes.getrow(i).toarray()
    snr.append(10*log10(mean(x_notes.getrow(i).toarray()**2)/mean(noise**2)))
print '{} +- {}'.format(mean(snr), sqrt(var(snr)))
hist(snr, bins=20)
pass

# <codecell>


# <codecell>


