''' Useful utils for BP-NMF

CREATED: 2013-05-02 05:14:35 by Dawen Liang <dl2771@columbia.edu>    

'''

import functools, pickle, sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import bp_vbayes
import librosa

specshow_cm = functools.partial(plt.imshow, aspect='auto', origin='lower', interpolation='nearest')
specshow = functools.partial(plt.imshow, cmap=plt.cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

def logspec(X, amin=1e-10, dbdown=80):
    ''' Compute the spectrogram matrix from STFT matrix

    Required arguments:
        X:          F by T STFT matrix (numpy.ndarray)

    Optional arguments:
        amin:       minimum amplitude threshold

        dbdown:     the minimum db below the maximum value

    '''
    logX = 20 * np.log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)

def gsubplot(args=(), cmap=plt.cm.gray_r):
    ''' General subplot
    Plot all the components passed in vertically. Each element of the arguments should be either a dict with matrix as data and string as title or a matrix.
    '''
    nargs = len(args)
    for i in xrange(nargs):
        plt.subplot(nargs, 1, i + 1)
        if type(args[i]) == dict:
            specshow_cm(args[i]['D'], cmap=cmap)
            plt.title(args[i]['T'])
        else:
            specshow_cm(args[i], cmap=cmap)
        plt.colorbar()
    plt.tight_layout()
    return

def get_data(filename, n_fft, hop_length, reweight=False, amin=1e-10, dbdown=80):
    x, _ = librosa.load(filename)
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    print X.shape
    specshow(logspec(X))
    if reweight:
        std_col = np.maximum(np.sqrt(np.var(X, axis=1, keepdims=True)), amin)
    else:
        std_col = 1.
    X /= std_col
    # cut off values 80db below maximum for numerical consideration
    X = np.maximum(X, 10**(-dbdown/10)*X.max())
    return (X, std_col)

def gen_train_seq(is_timed, N):
    if is_timed:
        timed = np.ones((N,), dtype='bool')
        timed[0] = False
    else:
        timed = np.zeros((N,), dtype='bool')
    return timed

def gen_save_name(id, is_timed, reweight, good_k=None):
    str_scaled = 'Scale' if reweight else 'Nscale'
    str_timed = '1N19T' if is_timed else '20N'
    name = 'bnmf_{}_{}_{}'.format(id, str_scaled, str_timed)
    if good_k is not None:
        name += '_GK{}'.format(good_k)
    return name

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    pass

def load_object(filename):
    with open(filename, 'r') as output:
        obj = pickle.load(output)
    return obj 

def train(X, K, init_option, alpha, N, timed, objs=None, RSeed=np.random.seed(), bnmf=None):
    if bnmf is None:
        bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, RSeed=RSeed, alpha=alpha)
    for n in xrange(N):
        start_t = time.time()
        ind = bnmf.update(timed=timed[n])
        if ind == -1:
            if n <= 1:
                # the initialization can be bad and the first/second iteration will suck, so restart
                print '***Bad initial values, restart***'
                return train(X, K, init_option, alpha, N, timed, objs=objs, RSeed=RSeed)
            else:
                # this should rarely happen
                print '***Oops***'
                #sys.exit(-1)
        t = time.time() - start_t
        if objs is not None:
            objs[n] = bnmf.obj
        print 'Dictionary Learning: Iteration: {}, good K: {}, time: {:.2f}, obj: {}'.format(n, bnmf.good_k.shape[0], t, bnmf.obj)
    return bnmf

def encode(bnmf, X, K, ED, ED2, init_option, alpha, N, timed, RSeed=np.random.seed()):
    bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, encoding=True, RSeed=RSeed, alpha=alpha)
    bnmf.ED, bnmf.ED2 = ED, ED2
    for n in xrange(N):
        start_t = time.time()
        ind = bnmf.encode(timed=timed[n])
        if ind == -1:
            return encode(bnmf, X, K, ED, ED2, init_option, alpha, N, timed)
        t = time.time() - start_t
        print 'Encoding: Iteration: {}, good K: {}, time: {:.2f}'.format(n, bnmf.good_k.shape[0], t)
    return bnmf

def separate(Xc, W, H, n_fft, save=False):
    ''' Separate copmlex spectrogram into components via Wiener Filter
    '''
    xl = [] 
    den = np.dot(W, H)
    den[den==0] += 1e-8
    L = W.shape[1]
    for l in xrange(L): 
        XL = Xc * np.outer(W[:,l], H[l,:])/den
        xl.append(librosa.istft(XL, n_fft=n_fft))
    if save:
        sio.savemat('xl.mat', {'xl':np.array(xl)})
    return np.array(xl)
