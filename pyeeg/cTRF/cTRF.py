'''
Package for with functions used for fitting the cTRF backward and forward models.
Mikolaj Kegler
Imperial College London, 20.05.2018
Contact: mikolaj.kegler16@imperial.ac.uk
'''

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.signal as signal
from . import cTRF_utils as utils

def ridge_fit_SVD(XtX, XtY ,lambdas, forward=False):
    '''
    Fast implementation of regularized least square fitting of the complex
    model's coefficients.The idea is to perform a computationally heavy operation
    of computing eigenvalues and eigenvectors only once and then simply compute
    the coefficients for variety of regularization parameters (multiple fast operations).

    Inputs:
    - XtX - covariance matrix of design matrix X. Shape: [N x N]
    - XtY - covariance matrix of design matrix X and vector Y. Shape: [N x M]
    - lambdas - list of regularization parameters to be considered of length R

    Output:
    - coeff - array of models coefficients for each regularization parameter.
    Shape: [R x N x M]

    Notes:
    - For forward model, the output matrix is rectangular [timelags x channels]
    but for backward model it is a row vector of length [timelags*channels].
    To obtain the rectangular shape, each of those row vectors need to be
    reshaped accordingly.
    '''

    # Compute eigenvaluesa and eigenvectors of covariance matrix XtX
    S, V = linalg.eigh(XtX, overwrite_a=True, turbo=True)

    # Sort the eigenvalues
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:, s_ind]

    # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
    # and compute the average
    tol = np.finfo(float).eps
    r = sum(S > tol)
    S = S[0:r]
    V = V[:, 0:r]
    nl = np.mean(S)

    # Compute z
    z = np.dot(V.T,XtY)

    # Initialize empty list to store coefficient for different regularization parameters
    coeff = []

    # Compute coefficients for different regularization parameters
    for l in lambdas:
        coeff.append(np.dot(V, (z/(S[:, np.newaxis] + nl*l))))

    # Flip the coefficients of the forward model to reflect the timelags w.r.t.
    # to fundamental waveform
    # Note: In order to reconstruct EEG from fundamental waveform with this
    # model it needs to be flipped column-wise once again!!!
    if forward:
        return np.array(coeff)[:, ::-1, :]
    else:
        return np.array(coeff)
        

def design_matrix(eeg, feat, tlag, cmplx=True, forward=False, normalize=True):
    '''
    Custom traning function. Takes training eeg, fundamental waveform (Y)
    datasets, zscores (optional) and fits the backward or forward model.

    Input:
    - eeg_list - eeg data. List of numpy arrays with shape [T x N],
    where T - number of samples, N - number of recording channels.
    - Y_list - speech signal features (envelope, fundamental waveform etc.).
    List of numpy arrays with shape [T x 1],
    where T - number of samples (the same as in EEG).
    - tlag - timelag range to consider in samples. Two element list.
    [-100, 400] means one does want to consider timelags of -100 ms and 400 ms
    for 1kHz sampling rate.
    - complex - boolean. True if complex model is considered and coeff will have
    complex values. Otherwise False and coeff will be real-only.
    - forward model - boolean. True if forward model shall be built.
    False if backward.

    Output:
    coeff - list of model coefficients for each considered regularization parameter.
    '''

    # eeg and feat need to have the same number of samples.
    assert eeg.shape[0] == feat.shape[0]
    assert feat.shape[1] == 1
    assert len(tlag) == 2

    lag_width = tlag[1] - tlag[0]

    # If forward model is to be considered swap the names of eeg and feat
    # variables, as now the goal is to map FW to EEG
    if forward == True:
        eeg, feat = feat, eeg
        tlag = np.array(tlag)[::-1]
    else:
        tlag = np.array(tlag)*-1

    # Align Y, so that it is misaligned with eeg by tlag_width samples
    Y = feat[tlag[0]:tlag[1], :]

    # Apply hilbert transform to EEG data (if backward model)
    # or speech feature (if forward model)
    if cmplx:
        eeg = utils.fast_hilbert(eeg, axis=0)

    # Preallocate memory for the design matrix X
    X = np.zeros((Y.shape[0], int(lag_width*eeg.shape[1])), dtype=eeg.dtype)

    # Fill in the design matrix X
    for t in range(Y.shape[0]):
        X[t, :] = eeg[t:(t + lag_width), :].reshape(lag_width*eeg.shape[1])

    # If complex concatenate the real and imaginary parts columne-wise
    if cmplx:
        X = np.hstack((X.real, X.imag))

    # Standardize X and Y matrices
    if normalize:
        X = stats.zscore(X, axis=0)
        Y = stats.zscore(Y, axis=0)

    return X, Y


def get_cov_mat(eeg_list, feat_list, tlag, cmplx=True, forward=False, normalize=True, n_sub=1):
    '''
    Extract features (X,Y) for training from list of parts, concatenate them and compute covariance matrices

    Input:
    - eeg_list - eeg data. List of numpy arrays with shape [T x N],
    where T - number of samples, N - number of recording channels.
    - feat_list - speech signal features (envelope, fundamental waveform etc.).
    List of numpy arrays with shape [T x 1], where T - number of samples.
    - tlag - timelag range to consider in samples(!). Two element list. [-100, 400]
    means one does want to consider timelags of -100 ms and 400 ms for 1kHz sampling rate.
    - complex - boolean. True if complex model is considered and coeff will have
    complex values. Otherwise False and coeff will be real-only.
    - forward model - boolean. True if forward model shall be built.
    False if backward.
    - normalize - boolean. Zscore eeg and speech featrues? (Default: True)
    - n_sub - integer. Number of subjects to be used in training pooled models. 
    Required to compute normalization factors. (Default: 1 - subject-specific model)
    
    Output:
    -XtX, XtY - covariance matrices XtX, XtY
    '''
    
    assert len(eeg_list) == len(feat_list)
    assert len(tlag) == 2
    
    X = []
    Y = []

    for (eeg, feat) in zip(eeg_list, feat_list):
        X_tmp, Y_tmp = design_matrix(eeg, feat, tlag, cmplx, forward, normalize)
        X.append(X_tmp)
        Y.append(Y_tmp)

    X = np.vstack(X)
    Y = np.vstack(Y)

    # Compute covariance matrices XtX and XtY
    if n_sub == 1:
        XtX = np.dot(X.T, X)
        XtY = np.dot(X.T, Y)
    elif n_sub > 1:        
        norm_pool_factor = np.sqrt((n_sub*X.shape[0] - 1)/(n_sub*(X.shape[0] - 1)))
        XtX = np.dot(X.T, X)*norm_pool_factor
        XtY = np.dot(X.T, Y)*norm_pool_factor
    
    return XtX, XtY


def train(eeg_list, feat_list, tlag, cmplx=True, forward=False, lambdas=[0], normalize=True):
    '''
    Custom traning function.
    Takes training eeg, speech feature (Y) datasets zscores (optional) and fits the
    backward or forward (complex, optional) model.

    Input:
    - eeg_list - eeg data. List of numpy arrays with shape [T x N],
    where T - number of samples, N - number of recording channels.
    - feat_list - speech signal features (envelope, fundamental waveform etc.).
    List of numpy arrays with shape [T x 1], where T - number of samples.
    - tlag - timelag range to consider in samples. Two element list. [-100, 400]
    means one does want to consider timelags of -100 ms and 400 ms for 1kHz sampling rate.
    - complex - boolean. True if complex model is considered and coeff will have
    complex values. Otherwise False and coeff will be real-only.
    - forward model - boolean. True if forward model shall be built.
    False if backward.
    - lambdas - range of regularization parameters to be considered.
    If None lambdas = [0] means no regularization.
    - normalize - boolean. Zscore eeg and speech featrues? (Default: True)

    Output:
    -coeff - list of model coefficients for each considered
    regularization parameter.
    '''

    XtX, XtY = get_cov_mat(eeg_list, feat_list, tlag, cmplx, forward, normalize)

    # Fit the model using chosen set of regularization parameters
    coeff = ridge_fit_SVD(XtX, XtY, lambdas, forward)

    return coeff


def pooled_train():
    ### TODO
    # Idea get_cov matrices for each participant x condition, then sum them together following Octave's recipe
    pass