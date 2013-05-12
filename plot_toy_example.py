# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.io as sio

# <codecell>

d = sio.loadmat('toy.mat')
W = flipud(np.maximum(d['ED'].T, 1e-10))
H = flipud(np.maximum(d['ES'], 1e-10))
print W.shape, H.shape

# <codecell>

n_basis = W.shape[0]
figure(1)
bases = W.copy()
for i in range(n_basis):
    bases[i] = 1 + 10*log10(bases[i]) / (np.abs(10*log10(bases[i])).max())
plot(bases.T + np.arange(bases.shape[0]).reshape((1, -1)))
axis('tight')

yticks(flipud(np.arange(bases.shape[0])) + 0.5, range(1, bases.shape[0] + 2))
grid('on')
xlabel('frequency (Hz)')
freq_res = 22050/512
xticks(arange(0, 251, 50), freq_res * arange(0, 251, 50))
ylabel('component')
#savefig('toy_W.pdf')

figure(2)
activation = H.copy()
for i in range(n_basis):
    activation[i] = 0.5 + 0.8 * activation[i] / (np.abs(activation[i]).max())
plot(activation.T + np.arange(activation.shape[0]).reshape((1, -1)))
axis('tight')

yticks(flipud(np.arange(activation.shape[0])) + 0.5, range(1, activation.shape[0] + 1))
grid('on')
time_res = 256./22050
xlabel('time (sec)')
xticks(arange(0, 250, 50), around(time_res * arange(0, 250, 50), 1))
ylabel('component')
#savefig('toy_H.pdf')

# <codecell>


# <codecell>


