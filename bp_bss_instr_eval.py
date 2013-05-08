# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, time, functools, midi, pickle
import scipy.io as sio
import scipy.sparse
import bp_vbayes, librosa, utils

# <codecell>

reload(utils)
pickle_file = 'exp/transcription/bnmf_mix_Scale_20N_GK62'
bnmf = utils.load_object(pickle_file)

# <codecell>

n_fft = 512
hop_length = 512
x, sr = librosa.load('../data/mix_var5a22k.wav')
Xc = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

good_k = bnmf.good_k
idx_comp = zeros((1, 62))
idx_comp[0] = arange(62)
x_sep = utils.separate(Xc, bnmf.ED[:,good_k], (around(bnmf.EZ) * bnmf.ES)[good_k,:], n_fft, hop_length, idx_comp=idx_comp)

# <codecell>

#x_sep = librosa.istft(dot(bnmf.ED[:,good_k], (around(bnmf.EZ) * bnmf.ES)[good_k, :])*exp(1j * np.angle(Xc)), n_fft=n_fft, hop_length=hop_length)
x_sep.shape

# <codecell>

x_sep = sum(x_sep, axis=0)
sio.savemat('x3', {'x3':x_sep})

# <codecell>

exp(J)

# <codecell>


