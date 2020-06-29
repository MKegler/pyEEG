# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,wrong-import-position
"""
In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)
    - CCA (in cca.py)
    - ...

TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.

"""
import logging
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse, mem_check
from .vizu import get_spatial_colors
from scipy import linalg

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__.split('.')[0])

def _get_covmat(x,y):
    '''
    
    '''
    return np.dot(x.T,y)


def _ridge_fit_SVD(x, y, alpha=[0.], from_cov=False):
    '''
    Classic implementation that's quite fast. Based on the Octave's original code.
    '''
    # Compute covariance matrices
    if not from_cov:
        XtX = _get_covmat(x,x)
        XtY = _get_covmat(x,y)
    else:
        XtX = x[:]
        XtY = y[:]

    # Cast alpha in ndarray
    if isinstance(alpha, float):
        alpha = np.asarray([alpha])
    else:
        alpha = np.asarray(alpha)

    # Compute eigenvalues and eigenvectors of covariance matrix XtX
    S, V = linalg.eigh(XtX, overwrite_a=False, turbo=True)

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
    for l in alpha:
        coeff.append(np.dot(V, (z/(S[:, np.newaxis] + nl*l))))

    return np.stack(coeff, axis=-1)


def _corr_multifeat(yhat, ytrue, nchans):
    return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=nchans)


def _rmse_multifeat(yhat, ytrue,axis=0):
    return np.sqrt(np.mean((yhat-ytrue)**2, axis))    


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
        - Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data (i.e. after
        the method :meth:`TRFEstimator.fit` has been called).
        - Can fit on a list of multiple dataset, where we have a list of target Y and
        a single stimulus matrix of features X, then the computation is made such that
        the coefficients computed are similar to those obtained by concatenating all matrices

    Examples
    --------
    >>> trf = TRFEstimator(tmin=-0.5, tmax-1.2, srate=125)
    >>> x = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000, 2)
    >>> trf.fit(x, y, lagged=False)
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alpha=[0.], fit_intercept=True, mtype='Forward'):

        # Times reflect mismatch a -> b, where a - dependent, b - predicted
        # Negative timelags indicate a lagging behind b
        # Positive timelags indicate b lagging behind a
        # For example:
        # eeg -> env (tmin = -0.5, tmax = 0.1)
        # Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alpha = alpha
        self.mtype = mtype # Forward or backward. Required for formatting coefficients in get_coef
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None
 
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        self.XtX_ = None
        self.XtY_ = None


    def fill_lags(self):
        """Fill the lags attributes.

        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.
        
        """
        if self.tmin and self.tmax:
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[::-1] #pylint: disable=invalid-unary-operand-type
            #self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]
        

    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)"%(estimated_mem_usage/1024.**3, mem_check()))

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]
        
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1] # this include non-valid samples for now

        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:, self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=drop, filling=np.nan if drop else 0.)

        return X, y


    def fit(self, X, y, lagged=False, drop=True, feat_names=()):
        """Fit the TRF model.

        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
        y : ndarray (nsamples x nchans)
            EEG data
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        method : string {svd | svd2}
            Choice of fitting method. 
            svd -> _svd_regress - default implementation by Hugo
            svd2 -> _svd_regress2 - Miko's implementation used for ABRs (2x faster?)

        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        self.coef_ = _ridge_fit_SVD(X, y, self.alpha)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0, np.newaxis, :]
            self.coef_ = self.coef_[1:, :]

        self.fitted = True

        return self


    def get_coef(self):
        '''
        Format and return coefficients
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1,:]

        return betas


    def add_cov(self, X, y, lagged=False, drop=True, n_parts=1):
        '''
        Compute and add (with normalization factor) covariance matrices XtX, XtY
        For v. large population models when it's not possible to load all the data to memory at once.
        '''
        if isinstance(X, (list,tuple)) and n_parts > 1:
            assert len(X) == len(y)
            
            for part in range(len(X)):
                assert len(X[part]) == len(y[part])
                X_part, y_part = self.get_XY(X[part], y[part], lagged, drop)
                XtX = _get_covmat(X_part, X_part)
                XtY = _get_covmat(X_part, y_part)
                norm_pool_factor = np.sqrt((n_parts*X_part.shape[0] - 1)/(n_parts*(X_part.shape[0] - 1)))

                if self.XtX_ is None:
                    self.XtX_ = XtX*norm_pool_factor
                else:
                    self.XtX_ += XtX*norm_pool_factor
                    
                if self.XtY_ is None:
                    self.XtY_ = XtY*norm_pool_factor
                else:
                    self.XtY_ += XtY*norm_pool_factor

        else:
            X_part, y_part = self.get_XY(X, y, lagged, drop)
            self.XtX = _get_covmat(X_part, X_part)
            self.XtY = _get_covmat(X_part, y_part)
        
        return self


    def fit_from_cov(self, X=None, y=None, lagged=False, drop=True, overwrite=True, part_length=150.):
        '''
        Fit model from covariance matrices (handy for v. large data)
        '''
        # If X and y are not none, chop them into pieces, compute cov matrices and fit the model (memory efficient)
        if (X is not None) and (y is not None):
            if overwrite:
                self.clear_cov() # If overwrite, wipe clean

            part_length = int(part_length*self.srate) # seconds to samples

            assert X.shape[0] == y.shape[0]

            segments = [(part_length * i) + np.arange(part_length) for i in range(X.shape[0] // part_length)] # Indices making up segments

            if X.shape[0] % part_length > part_length/2:
                segments = segments + [np.arange(segments[-1][-1], X.shape[0])] # Create new segment from leftover data (if more than 1/2 part length)
            else:
                segments[-1] = np.concatenate((segments[-1], np.arange(segments[-1][-1], X.shape[0]))) # Add letover data to the last segment
        
            X = [X[segments[i],:] for i in range(len(segments))]
            y = [y[segments[i],:] for i in range(len(segments))]

            self.add_cov(X, y, lagged=False, drop=True, n_parts=len(segments))

        self.fit_intercept = False
        self.intercept_ = None
        self.coef_ = _ridge_fit_SVD(self.XtX_, self.XtY_, self.alpha, from_cov=True)
        self.fitted = True
        return self


    def clear_cov(self):
        '''
        Wipe clean / reset covariance matrices.
        '''
        print("Clearing saved covariance matrices...")
        self.XtX_ = None
        self.XtY_ = None
        return self


    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.

        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).

        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates

        Notes
        -----
        If the matrix onky has features in its column (not yet lagged), the lagged version
        of the feature matrix will be created on the fly (this might take some time if the matrix
        is large).

        """
        assert self.fitted, "Fit model first!"

        if self.fit_intercept:
            betas = np.concatenate((self.intercept_, self.coef_), axis=0)
        else:
            betas = self.coef_[:]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            # LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
            if self.fit_intercept: # Adding intercept feature:
                X = np.hstack([np.ones((len(X), 1)), X])
        
        # Do it for every alpha
        pred = np.stack([X.dot(betas[:,:,i]) for i in range(betas.shape[-1])], axis=-1)

        return pred # Shape T x Nchan x Alpha


    def score(self, Xtest, ytrue, scoring="corr"):
        """Compute a score of the model given true target and estimated target from Xtest.

        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse")

        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.stack([_corr_multifeat(yhat[...,a], ytrue, nchans=self.n_chans_) for a in range(len(self.alpha))], axis=-1)
        elif scoring == 'rmse':
            return np.stack([_rmse_multifeat(yhat[...,a], ytrue) for a in range(len(self.alpha))], axis=-1)
        else:
            raise NotImplementedError("Only correlation score is valid for now...")


    def xval_eval(self, X, y, n_splits=5, lagged=False, drop=True, train_full=True, verbose=True, segment_length=None, fit='direct'):
        '''
        Standard cross-validation

        ToDo:
        - add standard scaler / normalizer
        '''

        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError("Supply several alphas to TRF constructor to use this method.")

        if segment_length:
            segment_length = segment_length*self.srate # time to samples

        self.fill_lags()

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        kf = KFold(n_splits=n_splits)
        if segment_length:
            scores = []
        else:
            scores = np.zeros((n_splits, self.n_chans_, len(self.alpha)))
        
        for kfold, (train, test) in enumerate(kf.split(X)):
            if verbose: print("Training/Evaluating fold %d/%d"%(kfold+1, n_splits))

            if fit == 'from_cov': # Fit using trick with adding covariance matrices -> saves RAM
                self.fit_from_cov(X[train,:], y[train,:], overwrite=True, part_length=150.)
            else: # Fit directly -> faster, but uses more RAM
                self.fit(X[train,:], y[train,:])

            if segment_length: # Chop testing data into smaller pieces
                
                if (len(test) % segment_length) > 0: # Crop if there are some odd samples
                    test_crop = test[:-int(len(test) % segment_length)]  
                else:
                    test_crop = test[:]
                
                test_segments = test_crop.reshape(int(len(test_crop) / segment_length), -1) # Reshape to # segments x segment duration
                
                ccs = [self.score(X[test_segments[i],:], y[test_segments[i],:]) for i in range(test_segments.shape[0])] # Evaluate each segment
                
                scores.append(ccs)
            else: # Evaluate using the entire testing data
                scores[kfold, :] = self.score(X[test,:], y[test,:])

        if segment_length:
            scores = np.asarray(scores)

        if train_full: 
            print("Fitting full model...")
            if fit == 'from_cov': # Fit using trick with adding covariance matrices -> saves RAM
                self.fit_from_cov(X, y, overwrite=True, part_length=150.)
            else:
                self.fit(X, y)

        return scores


### Obsolete?


    def plot(self, feat_id=None, alpha_id=None, ax=None, spatial_colors=False, info=None, **kwargs):
        """Plot the TRF of the feature requested as a *butterfly* plot.

        Parameters
        ----------
        feat_id : list or int
            Index of the feature requested or list of features.
            Default is to use all features.
        ax : array of axes (flatten)
            list of subaxes
        **kwargs : **dict
            Parameters to pass to :func:`plt.subplots`

        Returns
        -------
        fig : figure
        """
        if isinstance(feat_id, int):
            feat_id = list(feat_id) # cast into list to be able to use min, len, etc...
            if ax is not None:
                fig = ax.figure
        if not feat_id:
            feat_id = range(self.n_feats_)
        if len(feat_id) > 1:
            if ax is not None:
                fig = ax[0].figure
        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) < self.n_feats_]), "Feat ids not in range"

        if ax is None:
            if 'figsize' not in kwargs.keys():
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), figsize=(plt.rcParams['figure.figsize'][0] * np.size(feat_id), plt.rcParams['figure.figsize'][1]), **kwargs)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), **kwargs)

        if spatial_colors:
            assert info is not None, "To use spatial colouring, you must supply raw.info instance"
            colors = get_spatial_colors(info)
            
        for k, feat in enumerate(feat_id):
            if len(feat_id) == 1:
                ax.plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax.set_title('TRF for {:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax.get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
            else:
                ax[k].plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax[k].set_title('{:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax[k].get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
                        
        return fig


    def __getitem__(self, feats):
        "Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...)"
        # Argument check
        if self.feat_names_ is None:
            if np.ndim(feats) > 0:
                assert isinstance(feats[0], int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = feats

            else:
                assert isinstance(feats, int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]), "an element in argument %s in not present in %s"%(feats, self.feat_names_)
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s"%(feats, self.feat_names_)
                indices = [self.feat_names_.index(feats)]
                feats = [feats]

        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax, srate=self.srate, alpha=self.alpha)
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = feats
        trf.n_feats_ = len(feats)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_

        return trf


    def __repr__(self):
        obj = """TRFEstimator(
            alpha=%s,
            fit_intercept=%s,
            srate=%s,
            tmin=%s,
            tmax=%s,
            n_feats=%s,
            n_chans=%s,
            n_lags=%s,
            features : %s
        )
        """%(self.alpha, self.fit_intercept, self.srate, self.tmin, self.tmax,
             self.n_feats_, self.n_chans_, len(self.lags) if self.lags is not None else None, str(self.feat_names_))
        return obj


    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ == self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax, srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf