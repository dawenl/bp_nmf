# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle
import bp_vbayes
import librosa
import scipy.io as sio
import utils

# <codecell>

## Data-preprocessing parameters
reweight = True
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
#IDs = ['flute']

for ID in IDs:
    full_path = '../data/{}_var5a.wav'.format(ID)
    print full_path
    X, _ = utils.get_data(full_path, n_fft, hop_length, reweight=reweight)
    bnmf = utils.train(X, K, init_option, alpha, N, timed)
    name = utils.gen_save_name(ID, is_timed, reweight, good_k=bnmf.good_k.shape[0])
    utils.save_object(bnmf, name)

# <codecell>

bnmf.good_k.shape[0]

# <codecell>

utils.save_object(bnmf, name)

# <codecell>


