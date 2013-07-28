""" Useful utils for BP-NMF

CREATED: 2013-05-02 05:14:35 by Dawen Liang <dl2771@columbia.edu>

"""

import functools
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

import bp_nmf
import librosa

# Shortcuts
fig = functools.partial(plt.figure, figsize=(16, 4))
specshow = functools.partial(plt.imshow, cmap=plt.cm.hot_r, aspect='auto',
                             origin='lower', interpolation='nearest')


def logspec(X, amin=1e-10, dbdown=80):
    ''' Compute the spectrogram matrix from STFT matrix

    '''
    logX = 20 * np.log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)


def gsubplot(args=(), cmap=plt.cm.gray_r):
    ''' General subplot

    Plot all the components passed in vertically. Each element of the arguments
    should be either a dict with matrix as data and string as title or a matrix

    '''
    nargs = len(args)
    for i in xrange(nargs):
        plt.subplot(nargs, 1, i + 1)
        if isinstance(args[i], dict):
            specshow(args[i]['D'], cmap=cmap)
            plt.title(args[i]['T'])
        else:
            specshow(args[i], cmap=cmap)
        plt.colorbar()
    plt.tight_layout()
    return


def load_data(filename, n_fft, hop_length, sr=22050, amin=1e-10, dbdown=80,
              disp=1):
    ''' Load data with the specific fft size, hop length and sample rate

    '''
    x, _ = librosa.load(filename, sr=sr)
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    if disp:
        specshow(logspec(X))
        plt.colorbar()
    # cut off values 80db below maximum for numerical consideration
    X = np.maximum(X, 10**(-dbdown/10)*X.max())
    return X


## save/load bp_nmf objects
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    pass


def load_object(filename):
    with open(filename, 'r') as output:
        obj = pickle.load(output)
    return obj


## generate a name for bp_nmf object
def gen_save_name(id, n_fft, hop_length, K, good_k=None):
    name = 'bpnmf_{}_F{}_H{}_K{}'.format(id, n_fft, hop_length, K)
    if good_k is not None:
        name += '_GK{}'.format(good_k)
    return name


def dict_learn(X, K, seed=None, update_D=True, threshold=0.0001, maxiter=50,
               plot_obj=False):
    bpnmf = bp_nmf.LVI_BP_NMF(X, K=K, seed=seed)
    old_obj = -np.inf
    old_good_k = -1
    objs = []
    for i in xrange(maxiter):
        start_t = time.time()
        if not bpnmf.update(update_D=update_D, disp=1):
            if i <= 1:
                # the initialization can be bad and the first/second iteration
                # will suck, so restart
                # this can be potentially fixed by doing L-BFGS on each
                # univariate optimization, but will be substantially slower
                print '***Bad initial values, restart***'
                return None
            else:
                # This should rarely happen
                # If this happens in the late iteration, then we are almost
                # done. Return may not be the best idea, we are still getting
                # something
                print '***Oops***'
                return bpnmf
        t = time.time() - start_t
        objs.append(bpnmf.obj)
        improvement = (bpnmf.obj - old_obj) / abs(bpnmf.obj)
        old_obj = bpnmf.obj
        print('Iteration: {}, good K: {}, time: {:.2f}, '
              'obj: {:.2f} (improvement: {:.5f})'.format(i, bpnmf.good_k.size,
                                                         t, bpnmf.obj,
                                                         improvement))
        if improvement < threshold and old_good_k == bpnmf.good_k.size:
            break
        old_good_k = bpnmf.good_k.size
    if plot_obj:
        plt.figure()
        plt.plot(objs)
    return bpnmf


## ------------ for blind source separation ---------- ##
def wiener_mask(W, H, idx=None, amin=1e-10):
    ''' Constrct the wiener filter mask given the dictionary and activation
    '''
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
    ''' Interpolate the mask by a factor of 2
    '''
    F, T = mask.shape
    mask_interp = np.zeros((2*F-1, T))
    mask_interp[::2,:] = mask
    mask_interp[1::2,:] = mask[:-1,:] + np.diff(mask, axis=0)/2
    return mask_interp


def envelope(x, n_fft, hop_length):
    ''' Get envlope for a clean monophonic signal
        Simply taking the mean power spectrum
    '''
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    env = np.mean(X**2, axis=0)
    return env


def midi2notes(filename):
    ''' Load a midi file and get the time (in seconds) of each NoteOn
    and NoteOff (NoteOn with 0 velocity) event
    Mainly for MIREX F0 estimation data
    Strong assumption has made to the input midi file: the first track
    with information, the second track with actual notes
    '''
    try:
        import midi
    except ImportError:
        print 'Warning: midi module not available'
        return None

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
