# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle
import bp_vbayes
import librosa
import scipy.io as sio
import utils
import scipy.stats

# <headingcell level=1>

# Generating Data

# <codecell>

## Data-preprocessing parameters
reweight = False
n_fft = 512
hop_length = 512

## Model parameters
is_timed = False

K = 512
init_option = 'Rand'
alpha = 2.
N = 20
timed = utils.gen_train_seq(is_timed, N)

# <codecell>

reload(utils)
IDs = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe', 'mix']

for ID in IDs:
    full_path = '../data/{}_var5a22k.wav'.format(ID)
    print full_path
    X, _ = utils.get_data(full_path, n_fft, hop_length, reweight=reweight)
    bnmf = utils.train(X, K, init_option, alpha, N, timed)
    name = utils.gen_save_name(ID, is_timed, reweight, n_fft, hop_length, good_k=bnmf.good_k.shape[0])
    utils.save_object(bnmf, name)

# <headingcell level=1>

# Experiments

# <codecell>

instruments = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe']
true_count = {'bassoon': 22, 'clarinet': 13, 'flute': 15, 'oboe': 15, 'horn': 8}
count = 0
tcount = 0
dict_count = {}
for inst in instruments:
    pickle_file = 'exp/transcription/bnmf_{}_Nscale_20N_F512_H512_K512'.format(inst)
    bnmf = utils.load_object(pickle_file)
    good_k = bnmf.good_k
    count += len(good_k)
    tcount += true_count[inst] 
    dict_count[inst] = len(good_k)

# <codecell>

D = np.zeros((257, tcount))
S = np.zeros((tcount, 2325))
base_idx = 0
for inst in instruments:
    pickle_file = 'exp/transcription/bnmf_{}_Nscale_20N_F512_H512_K512'.format(inst)
    bnmf = utils.load_object(pickle_file)
    good_k = bnmf.good_k
    idx = np.flipud(np.argsort(bnmf.Epi[good_k]))
    end_idx = base_idx + true_count[inst]
    D[:,base_idx:end_idx] = bnmf.ED[:,good_k[idx[:true_count[inst]]]]
    S[base_idx:end_idx,:] = (around(bnmf.EZ)*bnmf.ES)[good_k[idx[:true_count[inst]]],:]
    base_idx += true_count[inst]

# <codecell>

utils.specshow(utils.logspec(S))

# <codecell>

idx = zeros((tcount,), dtype='int32')
base_idx = 0
for inst in instruments:
    end_idx = base_idx + true_count[inst]
    idx[base_idx: end_idx] = base_idx + argsort(argmax(D[:,base_idx:end_idx], axis=0))
    base_idx += true_count[inst]
figure(figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')
tmpD = D.copy()
tmpD = tmpD / np.median(tmpD, axis=0)
utils.specshow(utils.logspec(tmpD[:128,idx]))
xval = cumsum(true_count.values())
xval = hstack((0, xval))
xval[:-1] = xval[:-1] + diff(xval)/2
xticks(xval, true_count.keys(), fontsize=15)

xval = np.roll(cumsum(true_count.values()), 1)
xval[0] = 0
for v in xval[1:]:
    axvline(x=v-.4, color='black')
ylabel('frequency (Hz)', fontsize=12)
freq_res = 22050/(512*2)
yticks(arange(0, 128, 25), freq_res * arange(0, 251, 50))
tight_layout()
pass
savefig('bases.pdf')

# <codecell>

pickle_file = 'exp/transcription/bnmf_mix_Nscale_20N_F512_H512_K512_GK29'
bnmf = utils.load_object(pickle_file)
good_k = bnmf.good_k
idx = np.flipud(np.argsort(bnmf.Epi[good_k]))
ED = bnmf.ED[:,good_k[idx]]
ES = (around(bnmf.EZ)*bnmf.ES)[good_k[idx],:]
figure()
utils.specshow(utils.logspec(ED))
figure()
idx = argsort(argmax(ED, axis=0))
utils.specshow(utils.logspec(ED[:,idx]))

# <codecell>

D_norm = D / np.sqrt(np.sum(D**2, axis=0, keepdims=True))
ED_norm = ED.copy()
ED_norm = ED_norm / np.sqrt(np.sum(ED_norm**2, axis=0, keepdims=True))
Cor = dot(ED_norm.T, D_norm)

figure(1)
utils.specshow(Cor)
colorbar()

idx_all = flipud(argsort(Cor, axis=None))
idx_x = idx_all / Cor.shape[1]
idx_y = idx_all % Cor.shape[1]

idx = -1*ones((ED.shape[1],), dtype='int32')
count = 0
ind = zeros((D.shape[1],), dtype='bool')
for i in xrange(len(idx_all)):
    if ind[idx_y[i]] == False and idx[idx_x[i]] == -1:
        idx[idx_x[i]] = idx_y[i]
        ind[idx_y[i]] = True
        count += 1
    if count == len(idx):
        break

# <codecell>

figure()
subS = S[idx,:]
utils.specshow(utils.logspec(subS))
figure()
mask_S = (ES != 0)
ind = (sum(mask_S, axis=1) > 0.05*ES.shape[1])
subS = subS[ind, :]
utils.specshow(utils.logspec(subS))
figure()
ES = ES[ind, :]
utils.specshow(utils.logspec(ES))

# <codecell>

ES_z = ES - np.mean(ES, axis=1, keepdims=True)
subS_z = subS - np.mean(subS, axis=1, keepdims=True)
cor = np.sum(ES_z * subS_z, axis=1) / np.sqrt(np.sum(ES_z**2, axis=1) * np.sum(subS_z**2, axis=1))

mcor = np.mean(cor)
vcor = np.sqrt(np.var(cor))
figure()
plot(flipud(sort(cor)) ,'-o')
print mcor, vcor, median(cor)
figure()
hist(cor, bins=20)
pass

# <codecell>

p = zeros((100,))
for i in xrange(100):
    ridx = random.choice(S.shape[0], size=ES.shape[0], replace=False)
    rsubS = S[ridx,:]
    rsubS_z = rsubS - np.mean(rsubS, axis=1, keepdims=True)
    rcor = np.sum(ES_z * rsubS_z, axis=1) / np.sqrt(np.sum(ES_z**2, axis=1) * np.sum(rsubS_z**2, axis=1))
    p[i] = scipy.stats.wilcoxon(cor, rcor)[1]
print mean(p), sqrt(var(p))

# <codecell>

figure(figsize=(8,2))
boxplot([rcor, cor], 0,'rs',0, widths=.5)
#xlim([-0.1, 1.05])
xlabel('correlation', fontsize=15)
yticks([2,1], ['BP-NMF\n match', 'Random'], fontsize=15)
tight_layout()
pass
#savefig('trans.pdf')

# <codecell>


