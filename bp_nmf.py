# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import bp_vbayes
import librosa

# <codecell>

x, sr = librosa.load('../data/dawen.wav', sr=8000)
X = abs(librosa.stft(x, n_fft=512))
X = X/mean(X)

# <codecell>

reload(bp_vbayes)
bnmf = bp_vbayes.BpNMF(X, K=512)

# <codecell>

bnmf.update()

# <codecell>

subplot(211)
imshow(X, cmap=cm.gray_r, aspect='auto', origin='lower', interpolation='nearest')
colorbar()
subplot(212)
imshow(X - dot(bnmf.ED, bnmf.ES * bnmf.EZ), cmap=cm.gray_r, aspect='auto', origin='lower', interpolation='nearest')
colorbar()


# <codecell>

bnmf.ES * bnmf.EZ

# <codecell>

plot(sort(bnmf.Epi))

# <codecell>

exp(p0[361])

# <codecell>

(p1 >= p0).astype(int)

# <codecell>


