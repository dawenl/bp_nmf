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
    
n_notes = 0
for (inst, note) in notes.items():
    n_notes += len(note)

# <headingcell level=1>

# Signal for each distinct note

# <codecell>

base_idx = 0
data = []
row = []
col = []

for inst in instruments:
    x, sr = librosa.load('../data/{}_var5a22k.wav'.format(inst))
    num_samples = len(x)               
    notes_inst = notes[inst]
    for i, key in enumerate(notes_inst.keys()):
        n_event = len(notes_inst[key])/2
        for j in xrange(n_event):
            dur = (np.arange(round(sr * notes_inst[key][2*j]), round(sr * notes_inst[key][2*j+1])+1)).astype(int)
            data.extend(x[dur])
            col.extend(dur)
        row.extend(len(data_row) * [base_idx + i])
    base_idx += len(notes_inst)
x_notes = scipy.sparse.csr_matrix(scipy.sparse.coo_matrix((data, (row, col)), shape=(n_notes, num_samples)))

# <headingcell level=1>

# Power Envelope for each distinct note

# <codecell>

n_fft = 512
hop_length = 512
n_frames = 1 + int( (num_samples - n_fft) / hop_length)
X_envelope = np.zeros((n_notes, n_frames))

for i in xrange(n_notes):
    X_envelope[i,:] = utils.envelope((x_notes.getrow(i).T).todense(), n_fft, hop_length)
X_envelope = (X_envelope - np.mean(X_envelope, axis=1, keepdims=True)) / np.sqrt(np.var(X_envelope, axis=1, keepdims=True))

# <headingcell level=1>

# Load the model and evaluate

# <codecell>

pickle_file = 'exp/transcription/bnmf_mix_Nscale_20N_GK18'
bnmf = utils.load_object(pickle_file)
x, sr = librosa.load('../data/mix_var5a22k.wav')
num_samples = len(x)

# <codecell>

good_k = bnmf.good_k
idx = flipud(argsort(bnmf.Epi[good_k]))

#tmpES = (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:]
# L_inf normalization
#tmpES *= np.max(bnmf.ED[:,good_k[idx]], axis=0, keepdims=True).T
# L_2 normalization
#tmpES *= np.sum(bnmf.ED[:,good_k[idx]]**2, axis=0, keepdims=True).T**0.5

# calcualte the correlation
tmpES = (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:]
tmpES = (tmpES - np.mean(tmpES, axis=1, keepdims=True)) / np.sqrt(np.var(tmpES, axis=1, keepdims=True))

# <codecell>

multi2one = False
if multi2one:
    idx_comp = []
    count_comp = []
    for i in xrange(n_notes):
        tmp = dot(tmpES, X_envelope[i,:])
        bound = len(good_k)-1-np.argmax(np.diff((np.sort(tmp))))
        tmp_idx = flipud(np.argsort(dot(tmpES, X_envelope[i,:])))
        idx_comp.append(tmp_idx[:bound])
        count_comp.extend(tmp_idx[:bound])
        #figure(i)
        #plot((dot(tmpES, X_envelope[i,:]))[tmp_idx], '-bx')
        #plot((dot(tmpES, X_envelope[i,:]))[tmp_idx[:bound]], '-ro')
else:
    idx_comp = np.argmax(dot(tmpES, X_envelope.T), axis=0)
figure(1)
hist(np.argmax(dot(tmpES, X_envelope.T), axis=0), bins=len(good_k))
#figure(2)
#hist(count_comp, bins=len(good_k))
pass

# <codecell>

x_sep = []
for i in xrange(n_notes):
    mask = utils.wiener_mask(bnmf.ED[:,good_k[idx]], (around(bnmf.EZ) * bnmf.ES)[good_k[idx],:], idx=idx_comp[i])
    window = zeros((2*n_fft,))
    window[:n_fft] = 1
    X_ola = utils.stft(x, n_fft=2*n_fft, hann_w=window, hop_length=hop_length)
    mask_interp = utils.interp_mask(mask)
    x_sep.append(utils.istft(X_ola * mask_interp[:,:X_ola.shape[1]], n_fft=2*n_fft, hop_length=hop_length, hann_w=window))
x_sep = np.array(x_sep)

# <codecell>

x_res = np.zeros((n_notes, num_samples))
x_res[:, :x_sep.shape[1]] = x_sep
del x_sep

# <codecell>

snr = []
for i in xrange(n_notes):
    noise = x_res[i,:] - x_notes.getrow(i).toarray()
    snr.append(10*log10(mean(x_notes.getrow(i).toarray()**2)/mean(noise**2)))
print 'Average SNR: {} +- {}'.format(mean(snr), sqrt(var(snr)))
hist(snr, bins=20)
pass

# <codecell>

sio.savemat('x_res.mat', {'x_res':x_res})

# <codecell>


