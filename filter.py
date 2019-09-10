import numpy as np
from scipy.signal import fftconvolve

#-----------------------------------------------------------------------------------------------
def baxterking_filter(X, low=6, high=32, K=12):
    """
    Baxter-King bandpass filter

    Parameters
    ----------
    X : array-like
        A 1 or 2d ndarray. If 2d, variables are assumed to be in columns.
    low : float
        Minimum period for oscillations, ie., Baxter and King suggest that
        the Burns-Mitchell U.S. business cycle has 6 for quarterly data and
        1.5 for annual data.
    high : float
        Maximum period for oscillations BK suggest that the U.S.
        business cycle has 32 for quarterly data and 8 for annual data.
    K : int
        Lead-lag length of the filter. Baxter and King propose a truncation
        length of 12 for quarterly data and 3 for annual data.

    Returns
    -------
    Y : array
        Cyclical component of X

    Notes
    -----
    Returns a centered weighted moving average of the original series. Where
    the weights a[j] are computed ::

      a[j] = b[j] + theta, for j = 0, +/-1, +/-2, ... +/- K
      b[0] = (omega_2 - omega_1)/pi
      b[j] = 1/(pi*j)(sin(omega_2*j)-sin(omega_1*j), for j = +/-1, +/-2,...

    and theta is a normalizing constant ::

      theta = -sum(b)/(2K+1)
    """
    if type(x) == np.ndarray:
            x = x
    else:
        x = x.values
    X = np.asarray(X)

    # convert from freq. to periodicity
    omega_1 = 2.*np.pi/high
    omega_2 = 2.*np.pi/low
    bweights = np.zeros(2*K+1)
    # weight at zero freq.
    bweights[K] = (omega_2 - omega_1)/np.pi
    j = np.arange(1,int(K)+1)
    weights = 1/(np.pi*j)*(np.sin(omega_2*j)-np.sin(omega_1*j))
    # j is an idx
    bweights[K+j] = weights

    # make symmetric weights
    bweights[:K] = weights[::-1]
    # make sure weights sum to zero
    bweights -= bweights.mean()
    if X.ndim == 2:
        bweights = bweights[:,None]

    # get a centered moving avg/# convolution
    X = fftconvolve(X, bweights, mode='valid')
    return X
#-----------------------------------------------------------------------------------------------
def hamilton_filter(data, h, *args):
    """
    This function applies "Hamilton filter" to the data

    Parameters
    ----------
    data : arrray or dataframe
    h : integer
        Time horizon that we are likely to predict incorrectly.
        Original paper recommends 2 for annual data, 8 for quarterly data,
        24 for monthly data.
    *args : integer
        If supplied, it is p in the paper. Number of lags in regression.
        Must be greater than h.
        If not supplied, random walk process is assumed.

    Note: For seasonal data, it's desirable for p and h to be integer multiples
          of the number of obsevations in a year.
          e.g. For quarterly data, h = 8 and p = 4 are recommended.
          
    Returns
    -------
    cycle : array of cyclical component
    trend : trend component
    """
    if type(data) == np.ndarray:
            data = data
    else:
        data = data.values
    # transform data to array
    y = np.asarray(data, float)
    # sample size
    T = len(y)

    if len(args) == 1: # if p is supplied
        p = args[0]
        # construct X matrix of lags
        X = np.ones((T-p-h+1, p+1))
        for j in range(1, p+1):
            X[:, j] = y[p-j:T-h-j+1:1]

        # do OLS regression
        b = np.linalg.solve(X.transpose()@X, X.transpose()@y[p+h-1:T])
        # trend component (`nan` for the first p+h-1 period)
        trend = np.append(np.zeros(p+h-1)+np.nan, X@b)
        # cyclical component
        cycle = y - trend

    elif len(args) == 0: # if p is not supplied (random walk)
        cycle = np.append(np.zeros(h)+np.nan, y[h:T] - y[0:T-h])
        trend = y - cycle

    return cycle, trend
#-----------------------------------------------------------------------------------------------
def hodrickprescott_filter(X, λ=1600):
    """
    Hodrick-Prescott filter
    Parameters
    ----------
    X : array-like
        The 1d ndarray timeseries to filter of length (nobs,) or (nobs,1)
    λ : float
        The Hodrick-Prescott smoothing parameter. A value of 1600 is
        suggested for quarterly data, 6.25 for annual data and 129600 for monthly
        data.
    Returns
    -------
    cycle : array
        The estimated cycle in the data given lamb.
    trend : array
        The estimated trend in the data given lamb.
    -----
    The HP filter removes a smooth trend, `T`, from the data `X`. by solving
    min sum((X[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
     T   t
    The solution can be written as
    T = inv(I - lamb*K'K)X
    where I is a nobs x nobs identity matrix, and K is a (nobs-2) x nobs matrix
    such that
        K[i,j] = 1 if i == j or i == j + 2
        K[i,j] = -2 if i == j + 1
        K[i,j] = 0 otherwise
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    if type(X) == np.ndarray:
            X = X
    else:
        X = X.values

    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)

    I = sparse.eye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = sparse.dia_matrix((data, offsets), shape=(nobs-2,nobs))

    trend = spsolve(I+λ*K.T.dot(K), X, use_umfpack=True)

    cycle = X-trend
    return cycle, trend
#-----------------------------------------------------------------------------------------------
def christianofitzgerald_filter(X, low=6, high=32, drift=True):
    """
    Christiano Fitzgerald asymmetric, random walk filter

    Parameters
    ----------
    X : array-like
        1 or 2d array to filter. If 2d, variables are assumed to be in columns.
    low : float
        Minimum period of oscillations. Features below low periodicity are
        filtered out. Default is 6 for quarterly data, giving a 1.5 year
        periodicity.
    high : float
        Maximum period of oscillations. Features above high periodicity are
        filtered out. Default is 32 for quarterly data, giving an 8 year
        periodicity.
    drift : bool
        Whether or not to remove a trend from the data. The trend is estimated
        as np.arange(nobs)*(X[-1] - X[0])/(len(X)-1)

    Returns
    -------
    cycle : array
        The features of `X` between periodicities given by low and high
    trend : array
        The trend in the data with the cycles removed.

    """
    if low < 2:
        raise ValueError("low must be >= 2")

    if type(X) == np.ndarray:
            X = X
    else:
        X = X.values

    X = np.asanyarray(X)
    if X.ndim == 1:
        X = X[:,None]

    nobs, nseries = X.shape
    a = 2*np.pi/high
    b = 2*np.pi/low

    if drift: # get drift adjusted series
        X = X - np.arange(nobs)[:,None]*(X[-1] - X[0])/(nobs-1)

    J = np.arange(1,nobs+1)
    Bj = (np.sin(b*J)-np.sin(a*J))/(np.pi*J)
    B0 = (b-a)/np.pi
    Bj = np.r_[B0,Bj][:,None]
    y = np.zeros((nobs,nseries))

    for i in range(nobs):

        B = -.5*Bj[0] -np.sum(Bj[1:-i-2])
        A = -Bj[0] - np.sum(Bj[1:-i-2]) - np.sum(Bj[1:i]) - B
        y[i] = Bj[0] * X[i] + np.dot(Bj[1:-i-2].T,X[i+1:-1]) + B*X[-1] + \
                np.dot(Bj[1:i].T, X[1:i][::-1]) + A*X[0]
    y = y.squeeze()

    cycle, trend = y, X.squeeze()-y

    return cycle, trend
