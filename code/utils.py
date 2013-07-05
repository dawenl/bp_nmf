''' Useful utils for BP-NMF

CREATED: 2013-05-02 05:14:35 by Dawen Liang <dl2771@columbia.edu>    

'''

import functools, pickle, time
import numpy as np
import matplotlib.pyplot as plt

import bp_vbayes
import librosa

try:
    import midi
except ImportError:
    print 'Warning: midi module not available' 

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
        if type(args[i]) is dict:
            specshow(args[i]['D'], cmap=cmap)
            plt.title(args[i]['T'])
        else:
            specshow(args[i], cmap=cmap)
        plt.colorbar()
    plt.tight_layout()
    return

def get_data(filename, n_fft, hop_length, sr=22050, reweight=False, amin=1e-10, dbdown=80):
    x, _ = librosa.load(filename, sr=sr)
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    specshow(logspec(X))
    if reweight:
        std_col = np.maximum(np.sqrt(np.var(X, axis=1, keepdims=True)), amin)
    else:
        std_col = 1.
    X /= std_col
    # cut off values 80db below maximum for numerical consideration
    X = np.maximum(X, 10**(-dbdown/10)*X.max())
    return (X, std_col)

def gen_save_name(id, reweight, n_fft, hop_length, K, good_k=None):
    str_scaled = 'Scale' if reweight else 'Nscale'
    name = 'bnmf_{}_{}_F{}_H{}_K{}'.format(id, str_scaled, n_fft, hop_length, K)
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

def train(X, K, init_option, alpha, N, objs=None, RSeed=np.random.seed(),
        bnmf=None, multi_D=False, multi_S=True, disp=0):
    if bnmf is None:
        bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, RSeed=RSeed, alpha=alpha)
    for n in xrange(N):
        start_t = time.time()
        ind = bnmf.update(multi_D=multi_D, multi_S=multi_S, disp=disp)
        if not ind :
            if n <= 1:
                # the initialization can be bad and the first/second iteration will suck, so restart
                print '***Bad initial values, restart***'
                return train(X, K, init_option, alpha, N, objs=objs, RSeed=RSeed, multi_D=multi_D, multi_S=multi_S, disp=disp)
            else:
                # this should rarely happen
                print '***Oops***'
                #sys.exit(-1)
        t = time.time() - start_t
        if objs is not None:
            objs[n] = bnmf.obj
        print 'Dictionary Learning: Iteration: {}, good K: {}, time: {:.2f}, obj: {}'.format(n, bnmf.good_k.shape[0], t, bnmf.obj)
    return bnmf

def encode(bnmf, X, K, ED, ED2, init_option, alpha, N, RSeed=np.random.seed(), fmin='LBFGS'):
    bnmf = bp_vbayes.Bp_NMF(X, K=K, init_option=init_option, encoding=True, RSeed=RSeed, alpha=alpha)
    bnmf.ED, bnmf.ED2 = ED, ED2
    for n in xrange(N):
        start_t = time.time()
        ind = bnmf.encode(fmin=fmin)
        if not ind:
            return encode(bnmf, X, K, ED, ED2, init_option, alpha, N, fmin=fmin)
        t = time.time() - start_t
        print 'Encoding: Iteration: {}, good K: {}, time: {:.2f}, obj: {}'.format(n, bnmf.good_k.shape[0], t, bnmf.obj)
    return bnmf

def wiener_mask(W, H, idx=None, amin=1e-10):
    X_rec = np.maximum(np.dot(W, H), amin)
    if idx is None:
        L = W.shape[1]
        idx = np.arange(L)
    if len(np.shape(idx)) == 0:
        print 'Info: Separate out single component'
        mask = np.maximum(np.outer(W[:, idx], H[idx, :]), amin)
    elif len(np.shape(idx)) == 1:
        print 'Info: Separate out mixed components'
        mask = np.maximum(np.dot(W[:, idx], H[idx, :]), amin)
    else:
        print 'Error: Oops, wrong idx_comp dimension'
        return None

    mask = mask / X_rec
    return mask

def interp_mask(mask):
    F, T = mask.shape
    mask_interp = np.zeros((2*F-1, T))
    mask_interp[::2,:] = mask
    mask_interp[1::2,:] = mask[:-1,:] + np.diff(mask, axis=0)/2
    return mask_interp

def envelope(x, n_fft, hop_length):
    '''Get envlope for a clean monophonic signal
    '''
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    env = np.mean(X**2, axis=0)
    return env

def midi2notes(filename):
    '''Load a midi file and get the time (in seconds) of each NoteOn and NoteOff (NoteOn with 0 velocity) event
    Only for source separation purposes
    Strong assumption has made to the input midi file: the first track with information, the second track with actual notes
    '''
    f = midi.read_midifile(filename)
    f.make_ticks_abs()
    info_track = f[0]
    bpm = None
    for event in info_track:
        if isinstance(event, midi.SetTempoEvent):
            bpm = event.bpm
    if bpm is None:
        bpm = 120.
    tick_scale = 60./(bpm * f.resolution)    # second/tick
    
    note_track = f[1]
    notes = {}
    for event in note_track:
        if isinstance(event, midi.NoteOnEvent):
            if event.pitch not in notes:
                notes[event.pitch] = []
                assert(event.velocity != 0)
                notes[event.pitch].append(event.tick * tick_scale)
            else:
                notes[event.pitch].append(event.tick * tick_scale)
    # sanity check
    for _, sth in notes.items():
        assert(len(sth) % 2 == 0)
    return notes
