# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools
import gap_nmf, librosa

import numpy as np

from matplotlib.pyplot import *

# <codecell>

fig = functools.partial(figure, figsize=(16, 4))
specshow = functools.partial(imshow, cmap=cm.jet, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

x, _ = librosa.load('path_to_some_music', sr=44100)
X = np.abs(librosa.stft(x, n_fft=512, hop_length=256))

# <codecell>

obj = gap_nmf.GaP_NMF(X, K=50, seed=98765)

# <codecell>

score = -np.inf
criterion = 0.0005
for i in xrange(1000):
    obj.update()
    obj.figures()
    lastscore = score
    score = obj.bound()
    improvement = (score - lastscore) / np.abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    if improvement < criterion:
        break

# <codecell>


