import numpy as np
from scipy import signal
from scipy import fftpack

def butter_lowpass_filter(data, cutoff, fs, order=1):
    '''
    Butterworth lowpass filter. SciPy model implementation.
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    '''
    Butterworth Bandpass filter. SciPy model implementation.
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def fast_hilbert(X, axis=0):
    '''
    Fast implementation of Hilbert transform. The trick is to find the next fast
    length of vector for fourier transform (fftpack.helper.next_fast_len(...)).
    Next the matrix of zeros of the next fast length is preallocated and filled
    with the values from the original matrix.

    Inputs:
    - X - input matrix
    - axis - axis along which the hilbert transform should be computed

    Output:
    - X - analytic signal of matrix X (the same shape, but dtype changes to np.complex)
    '''
    # Add dimension if X is a vector
    if len(X.shape) == 1:
        X = X[:,np.newaxis]

    fast_shape = np.array([fftpack.helper.next_fast_len(X.shape[0]), X.shape[1]])
    X_padded = np.zeros(fast_shape)
    X_padded[:X.shape[0], :] = X
    X = signal.hilbert(X_padded, axis=axis)[:X.shape[0], :]
    return X

def get_env(audio, bpf=[1, 10], resample=True, us=1, ds=480):
    '''
    Obtain speech envelope from analytical signal & filter
    '''
    env = np.abs(fast_hilbert(audio)) # Envelope extraciton
    env = np.squeeze(env)
    if len(bpf) > 1:
        env = butter_bandpass_filter(env, bpf[0], bpf[1], 48000, 1) # Bandpass filtering (butterworth, zero-phase)
    else:
        env = butter_lowpass_filter(env, bpf[0], 48000, 1) # Lowpass filtering (butterworth, zero-phase)
    if resample == True:
        env = signal.resample_poly(env, us, ds) # Downsample to 100 Hz
    return env

def get_FW(audio, bpf=[100, 300], resample=True, us=1, ds=48):
    '''
    Obtain fundamental waveform by filtering raw audio (100-300 default)
    '''
    if len(bpf) > 1:
        FW = butter_bandpass_filter(audio, bpf[0], bpf[1], 48000, 1) # Bandpass filtering (butterworth, zero-phase)
    else:
        FW = butter_lowpass_filter(audio, bpf[0], 48000, 1) # Lowpass filtering (butterworth, zero-phase)
    if resample == True:
        FW = signal.resample_poly(FW, us, ds) # Downsample to 1000 Hz
    return FW

def interp_eeg(eeg, t_len):
    '''
    Interpolate EEG to compensate for drifts
    '''
    x = np.arange(eeg.shape[0])
    x_intp = np.arange(t_len)
    eeg_intp = []
    for ch in eeg.T:
        ch_intp = np.interp(x_intp, x, ch)
        eeg_intp.append(ch_intp)
        del ch
    return np.array(eeg_intp).T