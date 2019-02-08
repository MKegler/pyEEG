# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:09:24 2019

@author: Karen
"""

import logging
import mne
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from mne.decoding import BaseEstimator
from pyeeg.utils import lag_matrix, lag_span, lag_sparse, is_pos_def, find_knee_point
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def cca_nt(x, y, thresh, knee_point):
    # A, B: transform matrices
    # R: r scores
    # can normalise the data
    # Build covariance matrix: C=[x,y]'*[x,y]
    m = np.size(x,1)
    C = np.concatenate([x, y], axis=1).T @ np.concatenate([x, y], axis=1)
    # C = np.cov(np.transpose(x),np.transpose(y)) # normalised
    
    # PCA on X.T X to get sphering matrix A1 and on Y.T*Y for A2
    As = []
    Eigvals = []
    for C_temp in [C[:m, :m], C[m:, m:]]:
        Val, Vec = np.linalg.eigh(C_temp)               # get eigval & eigvec
        if not is_pos_def(C_temp):
            discard = np.argwhere(Val < 0)
            Val = Val[max(discard)[0]+1:]
            Vec = Vec[:,max(discard)[0]+1:]
        Val, Vec = Val[::-1], Vec[:, ::-1]
        if knee_point is not None:
            find_knee_point()
            print('knee_point used')
            
        keep = np.cumsum(Val)/sum(Val) <= thresh   # only keep components over certain percentage of variance
        topcs = Vec[:, keep]                        # corresponding vecs
        Val = Val[keep]
        exp = 1-1e-12
        Val = Val**exp
        As.append(topcs @ np.diag(np.sqrt(1/Val)))
        Eigvals.append(Val)
    A1, A2 = As
    eigvals_x, eigvals_y = Eigvals 
    
    # create new C = Amix.T*C*Amix
    AA = np.zeros((A1.shape[0] + A2.shape[0], A1.shape[1] + A2.shape[1]))
    AA[:A1.shape[0], :A1.shape[1]] = A1
    AA[A1.shape[0]:, A1.shape[1]:] = A2
    C = AA.T @ C @ AA
    
    N = np.min((np.size(A1,1), np.size(A2,1)))    # number of canonical components
    
    # PCA on Cnew
    Val, Vec = np.linalg.eigh(C)
    Val, Vec = Val[::-1], Vec[:, ::-1]
    
    A = A1 @ Vec[:np.size(A1,1),:N]*np.sqrt(2)      # keeping only N first PCs
    B = A2 @ Vec[np.size(A1,1):,:N]*np.sqrt(2)
    R = Val[:N] - 1
    
    return A1, A2, A, B, R, eigvals_x, eigvals_y

class CCA_Estimator(BaseEstimator):
    
    """Canonocal Correlation (CCA) Estimator Class.

    Attributes
    ----------
    lags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features
    Notes
    -----
    Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data

   """
        
    def __init__(self, times =(0.,), tmin=None, tmax=None, srate=1., fit_intercept=True):
        
        if tmin and tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(tmin, tmax, srate=srate)
            self.times = self.lags / srate
        else:
            self.lags = lag_sparse(times, srate)
            self.times = np.asarray(times)
            
        self.srate = srate
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.X = 0
        self.y = 0
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coefStim_ = None
        self.coefResponse_ = None
        self.score_ = None
        self.eigvals_x = None
        self.eigvals_y = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        
    
    def fit(self, X, y, thresh, knee_point=None, drop=True, feat_names=()):
        """ Fit CCA model.
        
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
        y : ndarray (nsamples x nchans)
            EEG data

        Returns
        -------
        coef_ : ndarray (nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """
        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1]
        if feat_names:
            self.feat_names_ = feat_names

        # Creating lag-matrix droping NaN values if necessary
        if drop:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=True)

            # Droping rows of NaN values in y
            if any(np.asarray(self.lags) < 0):
                drop_top = abs(min(self.lags))
                y = y[drop_top:, :]
            if any(np.asarray(self.lags) > 0):
                drop_bottom = abs(max(self.lags))
                y = y[:-drop_bottom, :]
        else:
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
        self.X = X
        self.y = y
        
        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])          
        
        A1, A2, A, B, R, eigvals_x, eigvals_y = cca_nt(X, y, thresh, knee_point)
        
        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = A[0, :]
            A = A[1:, :]
            
        self.coefStim_ = A
        self.coefResponse_ = B
        self.score_ = R
        self.eigvals_x = eigvals_x
        self.eigvals_y =eigvals_y
        
    def plot_time_filter(self, n_comp=1, feat_id=0):
        """Plot the TRF of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        plt.plot(self.times, self.coefStim_[:, :n_comp])
        if self.feat_names_:
            plt.title('TRF for {:s}'.format(self.feat_names_[feat_id]))
            
    def plot_spatial_filter(self, pos, comp=0, feat_id=0):
        """Plot the topo of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        mne.viz.plot_topomap(self.coefResponse_[:, comp], pos)
        
    def plot_corr(self, X, y, pos, comp=0):
        """Plot the correlation between the EEG component wavefor and the EEG channel waveform.
        Parameters
        ----------
        """   
        eeg_proj = y.dot(self.coefResponse_[:,comp])
        env_proj = X.dot(self.coefStim_[:, comp])
        
        r = np.zeros(64)
        for i in range(64):
            r[i] = np.corrcoef(y[:,i], eeg_proj)[0,1]
    
        cc_corr = np.corrcoef(eeg_proj, env_proj)[0,1]
        fig, ax = plt.subplots()
        im, _ = mne.viz.plot_topomap(r, pos, axes=ax, show=False)
        ax.set(title=r"CC #{:d} ($\rho$={:.3f})".format(comp+1, cc_corr))
        plt.colorbar(im)
        mne.viz.tight_layout()
        
    def plot_activation_map(self, X, y, pos, n_comp=1):
        s_hat = y @ self.coefResponse_
        sigma_eeg = y.T @ y
        sigma_reconstr = s_hat.T @ s_hat
        a_map = sigma_eeg @ self.coefResponse_ @ np.linalg.inv(sigma_reconstr)
        _, ax = plt.subplots(nrows=1, ncols=n_comp, figsize=(12,8))
        for c in range(n_comp):
            im, _ = mne.viz.plot_topomap(a_map[:, c], pos, axes=ax[c], show=False)
            ax[c].set(title=r"CC #{:d}".format(c + 1))

        
        
        
        