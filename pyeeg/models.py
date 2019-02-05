# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,wrong-import-position
"""
In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)
    - CCA
    - ?

TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.

"""

import logging
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def _svd_regress(x, y, alpha=0.):
    """Linear regression using svd.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats)
    y : ndarray (nsamples, nchans)

    Returns
    -------
    betas : ndarray (nfeats, nchans)
        Coefficients

    Raises
    ------
    ValueError
        If alpha < 0 (coefficient of L2 - regularization)
    NotImplementedError
        If len(alpha) > 1 (later will do several fits)

    Notes
    -----
    A warning is shown in the case where nfeats > nsamples, if so the user
    should rather use partial regression.
    """
    try:
        assert len(alpha) == 1, "Alpha cannot be an array"
    except AssertionError:
        raise NotImplementedError
    try:
        assert alpha >= 0, "Alpha must be positive"
    except AssertionError:
        raise ValueError

    [U, s, V] = np.linalg.svd(x, full_matrices=False)
    Uty = U.T @ y
    Vsreg = V @ np.diag(s/(s**2 + alpha))
    betas = Vsreg @ Uty
    return betas

class TRFEstimator(BaseEstimator):
    """Temporal Response Function (TRF) Estimator Class.

    This class allows to estimate TRF from a set of feature signals and an EEG dataset in the same fashion
    than ReceptiveFieldEstimator does in MNE.
    However, an arbitrary set of lags can be given. Namely, it can be used in two ways:
        - calling with `tmin` and tmax` arguments will compute lags spanning from `tmin` to `tmax`
        - with the `times` argument, one can request an arbitrary set of time lags at which to compute
        the coefficients of the TRF

    Attributes
    ----------
    lags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    use_reularisation : bool
        Whether ot not the Ridge regularisation is used to compute the TRF
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    fitted : bool
        True once the TRF has been fitted on EEG data
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
    Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data (i.e. after
    the method :meth:`TRFEstimator.fit` has been called).

    #TODO: add example
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1.,
                 alpha=0., fit_intercept=True):

        if tmin and tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(tmin, tmax, srate=srate)
            self.times = self.lags / srate
        else:
            self.lags = lag_sparse(times, srate)
            self.times = np.asarray(times)

        self.srate = srate
        self.alpha = alpha
        self.use_regularisation = alpha > 0.
        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None

    def fit(self, X, y, drop=True, feat_names=()):
        """Fit the TRF model.

        Parameters
        ----------
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

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Solving with svd or least square:
        if self.use_regularisation:
            # svd method:
            betas = _svd_regress(X, y, self.alpha)
        else:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        self.coef_ = np.reshape(betas, (len(self.lags), self.n_feats_, self.n_chans_))
        self.fitted = True

    def plot_single_feature(self, feat_id):
        """Plot the TRF of the feature requested as a *butterfly* plot.

        Parameter
        ---------
        feat_id : int
            Index of the feature requested
        """
        assert self.fitted, "Fit the model first!"
        plt.plot(self.times, self.coef_[:, feat_id, :])
        if self.feat_names_:
            plt.title('TRF for {:s}'.format(self.feat_names_[feat_id]))
