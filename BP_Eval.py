# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, time, functools
import scipy.io as sio
import scipy.stats
import librosa, bp_vbayes
import utils

# <codecell>

## Data pre-processing parameters
reload(utils)
ID = 'PinkMoon'
filename = '../data/{}.mp3'.format(ID)
n_fft = 512
hop_length = 512
reweight = False
X, std_col = utils.get_data(filename, n_fft, hop_length, reweight=reweight)
# Take the middle 4000 frames for cross-validation
F, L = X.shape
D = X[:, (L-4000)/2.:(L+4000)/2.]

## Model training parameters
K = 512
init_option = 'Rand'
alpha = 2.
N = 20

# <codecell>

def cv_band_expansion(is_timed):
    timed = utils.gen_train_seq(is_timed, N)
    score = []
    good_ks = []
    for i in xrange(5):
        idx = ones((4000,), dtype='bool')
        idx[i*800:(i+1)*800] = False
        D_test = D[:, ~idx]
        D_train = D[:, idx]
        bnmf = utils.train(D_train, K, init_option, alpha, N, timed)
        Dict, Dict2 = bnmf.ED[:,bnmf.good_k].copy(), bnmf.ED2[:,bnmf.good_k].copy()
        bnmf = utils.encode(bnmf, D_test[:F-192,:], Dict.shape[1], Dict[:F-192,:], Dict2[:F-192,:], init_option, alpha, N, timed)
        X_pred = dot(Dict[:,bnmf.good_k], (bnmf.ES * around(bnmf.EZ))[bnmf.good_k,:])
        sigma = sqrt( mean( (D_test[:F-192,:] - X_pred[:F-192,:])**2 ) )
        score.append(sum(scipy.stats.norm.logpdf(X_pred[:-192,:], loc=D_test[:-192,:], scale=sigma)))  
        good_ks.append(bnmf.good_k.shape[0])
    print good_ks
    print score
    name = utils.gen_save_name(ID, is_timed, reweight)
    sio.savemat('{}.mat'.format(name), {'goodks':good_ks, 'score':score})    

# <headingcell level=1>

# Band expansion experiemnt for time-dependent BP-NMF

# <codecell>

'''
Band expansion for time-dependent BP-NMF
'''
cv_band_expansion(True)

# <headingcell level=1>

# Band expansion experiment for i.i.d. BP-NMF

# <codecell>

'''
Band expansion for i.i.d. BP-NMF
'''
cv_band_expansion(False)

# <codecell>


