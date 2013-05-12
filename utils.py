''' Useful utils for BP-NMF

CREATED: 2013-05-02 05:14:35 by Dawen Liang <dl2771@columbia.edu>    

'''

import functools, midi, pickle, time
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.signal

import bp_vbayes
import librosa

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

def stft(y, n_fft=256, hann_w=None, hop_length=None):
    """Short-time fourier transform

    Arguments:
      y           -- (ndarray)  the input signal
      n_fft       -- (int)      number of FFT components  | default: 256
      hann_w      -- (int)      size of Hann window       | default: n_fft
      hop_length  -- (int)      number audio of frames 
                                between STFT columns      | default: hann_w / 2

    Returns D:
      D           -- (ndarray)  complex-valued STFT matrix

    """
    num_samples = len(y)

    if hann_w is None:
        hann_w = n_fft

    if np.shape(hann_w) == 0:
        if hann_w == 0:
            window = np.ones((n_fft,))
        else:
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w), 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')
    else:
        window = hann_w

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft / 2)

    n_specbins  = 1 + int(n_fft / 2)
    n_frames    = 1 + int( (num_samples - n_fft) / hop_length)

    # allocate output array
    stft_matrix = np.empty( (n_specbins, n_frames), dtype=np.complex)

    for i in xrange(n_frames):
        sample  = i * hop_length
        frame   = fft.fft(window * y[sample:(sample+n_fft)])

        # Conjugate here to match phase from DPWE code
        stft_matrix[:, i]  = frame[:n_specbins].conj()

    return stft_matrix

def istft(stft_matrix, n_fft=None, hann_w=None, hop_length=None):
    """
    Inverse short-time fourier transform

    Arguments:
      stft_matrix -- (ndarray)  STFT matrix from stft()
      n_fft       -- (int)      number of FFT components   | default: inferred
      hann_w      -- (int)      size of Hann window        | default: n_fft
      hop_length  -- (int)      audio frames between STFT                       
                                columns                    | default: hann_w / 2

    Returns y:
      y           -- (ndarray)  time domain signal reconstructed from d

    """

    # n = Number of stft frames
    n_frames    = stft_matrix.shape[1]

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[0] - 1)

    if hann_w is None:
        hann_w = n_fft
    if np.shape(hann_w) == 0:
        if hann_w == 0:
            window = np.ones(n_fft)
        else:
            #   magic number alert!
            #   2/3 scaling is to make stft(istft(.)) identity for 25% hop
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w) * 2.0 / 3.0, 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')
    else:
        window = hann_w

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = n_fft / 2

    y = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)
        ytmp    = window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    return y

def gsubplot(args=(), cmap=plt.cm.gray_r):
    ''' General subplot
    Plot all the components passed in vertically. Each element of the arguments should be either a dict with matrix as data and string as title or a matrix.
    '''
    nargs = len(args)
    for i in xrange(nargs):
        plt.subplot(nargs, 1, i + 1)
        if type(args[i]) == dict:
            specshow(args[i]['D'], cmap=cmap)
            plt.title(args[i]['T'])
        else:
            specshow(args[i], cmap=cmap)
        plt.colorbar()
    plt.tight_layout()
    return

def get_data(filename, n_fft, hop_length, sr=None, reweight=False, amin=1e-10, dbdown=80):
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

def gen_train_seq(is_timed, N):
    if is_timed:
        timed = np.ones((N,), dtype='bool')
        timed[0] = False
    else:
        timed = np.zeros((N,), dtype='bool')
    return timed

def gen_save_name(id, is_timed, reweight, n_fft, hop_length, K, good_k=None):
    str_scaled = 'Scale' if reweight else 'Nscale'
    str_timed = '1N19T' if is_timed else '20N'
    name = 'bnmf_{}_{}_{}_F{}_H{}_K{}'.format(id, str_scaled, str_timed, n_fft, hop_length, K)
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
    Strong assumption has made to the input midi file: the first track with time information, the second track with actual notes
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
