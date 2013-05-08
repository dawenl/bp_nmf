# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time, pickle
import bp_vbayes
import librosa
import scipy.io as sio
import utils

# <headingcell level=1>

# Generating Data

# <codecell>

## Data-preprocessing parameters
reweight = False
n_fft = 512
hop_length = 512

## Model parameters
is_timed = True

K = 512
init_option = 'Rand'
alpha = 2.
N = 20
timed = utils.gen_train_seq(is_timed, N)

# <codecell>

reload(utils)
IDs = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe', 'mix']
#IDs = ['clarinet', 'horn']

for ID in IDs:
    full_path = '../data/{}_var5a22k.wav'.format(ID)
    print full_path
    X, _ = utils.get_data(full_path, n_fft, hop_length, reweight=reweight)
    bnmf = utils.train(X, K, init_option, alpha, N, timed)
    name = utils.gen_save_name(ID, is_timed, reweight, good_k=bnmf.good_k.shape[0])
    utils.save_object(bnmf, name)

# <headingcell level=1>

# Experiments

# <codecell>

instruments = ['bassoon', 'clarinet', 'flute', 'horn', 'oboe', 'mix']
pickle_file = 'exp/transcription/bnmf_bassoon_Scale_20N_GK23'
bnmf = utils.load_object(pickle_file)

# <codecell>


# <codecell>


