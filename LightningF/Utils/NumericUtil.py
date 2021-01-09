import os
import copy
import numpy as np
from scipy.special import gammaln, digamma, psi


CTE_NORM = - 0.5 * np.log(2. * np.pi)


def convert_to_n0(n):
    """
    Convert count vector to vector of "greater than" counts.

    Parameters
    -------
    n : 1D array, size K
        each entry k represents the count of items assigned to comp k.

    Returns
    -------
    n0 : 1D array, size K
        each entry k gives the total count of items at index above k
        N0[k] = np.sum(N[k:])

    Example
    -------
    >> convertToN0([1., 3., 7., 2])
    [12, 9, 2]
    """
    n = np.asarray(n)
    n0 = np.zeros_like(n)
    n0[:-1] = n[::-1].cumsum()[::-1][1:]

    return n0


def e_log_beta(eta1, eta0):
    """
    # Calculate expected mixture weights E[ log \beta_k ]
    # Using copy() allows += without modifying ElogU
    :param eta1:
    :param eta0:
    :return:
                Elogbeta :
    """
    elogu, elog1mu = calc_beta_expectations(eta1, eta0)

    elogbeta = elogu.copy()
    elogbeta[1:] += elog1mu[:-1].cumsum()
    return elogbeta


def e_log_n(x, sigma2x, mu, sigma2):
    n = x.shape[0]
    k, d = mu.shape
    elog = np.zeros([n, k])
    elog += (- 0.5 * d * np.log(2. * np.pi) - 0.5 * np.log(np.prod(sigma2x, axis=1)))[:, None]

    # New Calculation
    sxi = 0.5 * np.reciprocal(sigma2x)
    for c in np.arange(k):
        elog[:, c] += - np.sum(sxi[:] * np.square(x[:] - mu[c]), axis=1)
        elog[:, c] += - np.sum(sxi[:] * sigma2[c], axis=1)

    # Old Calculation
    # elog += - np.sum(0.5 * (sigmaX[:, None, :] ** -1) * (X[:, None, :] - mu[None, :, :]) ** 2, axis=2)
    # elog += - np.sum(0.5 * (sigma2[None, :, :]) * (sigmaX[:, None, :] ** -1), axis=2)

    return elog


def calc_beta_expectations(eta1, eta0):
    """
    Evaluate expected value of log u under Beta(u | eta1, eta0)
    :param eta1:
    :param eta0:
    :return:    ElogU : 1D array, size K
                Elog1mU : 1D array, size K

    """
    digamma_both = digamma(eta0 + eta1)
    elogu = digamma(eta1) - digamma_both
    elog1mu = digamma(eta0) - digamma_both
    return elogu, elog1mu


def inplace_exp_normalize_rows_numpy(r, clip=False):
    """ Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    """
    r -= np.max(r, axis=1)[:, np.newaxis]
    if clip:
        np.clip(r, -7., 7., out=r)
    np.exp(r, out=r)
    r /= r.sum(axis=1)[:, np.newaxis]


try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        # Scipy changed location of BLAS libraries in late 2012.
        # See http://github.com/scipy/scipy/pull/358
        fblas = scipy.linalg.blas._fblas
except:
    raise ImportError(
        "BLAS libraries for efficient matrix multiplication not found")


def dotatb(a, b):
    """ Compute matrix product A.T * B
        using efficient BLAS routines (low-level machine code)
    """
    if a.shape[1] > b.shape[1]:
        return fblas.dgemm(1.0, a, b, trans_a=True)
    else:
        return np.dot(a.T, b)


def delta_c(foo, km, k1, k2):

    if len(prior) == 2:
        calc = - foo(km[0], km[1]) + foo(k1[0], k1[1]) + foo(k2[0], k2[1])
    else:
        calc = - foo(km[0]) + foo(k1[0]) + foo(k2[0])

    return calc


def c_beta(eta1, eta0):
    return np.sum(gammaln(eta1 + eta0) - gammaln(eta1) - gammaln(eta0))


def c_dir(alphaij):
    return gammaln(alphaij.sum()) - gammaln(alphaij).sum()


def Lalloc(Nvec=None, SS=None, gamma=0.5, theta=None, Elogw=None):
    assert theta is not None
    K = theta.size
    if Elogw is None:
        Elogw = digamma(theta) - digamma(theta.sum())
    if Nvec is None:
        Nvec = SS.N
    Lalloc = c_dir(gamma * np.ones(K)) - c_dir(theta)
    Lalloc_slack = np.inner(Nvec + gamma - theta, Elogw)

    return Lalloc + Lalloc_slack


def c_h(mu, sigma2):

    c = CTE_NORM - 0.5 * np.log(sigma2) - 0.5 * np.square(mu) * sigma2**-1

    return np.sum(c)


def c_gamma(alpha, beta):
    return np.sum(alpha * np.log(beta) - gammaln(alpha))


def c_alpha(a, b, e_alpha):
    return a * np.log(b) - gammaln(a) - b * e_alpha


def elog_gamma(eta1, eta0):
    """
    :param eta1:
    :param eta0:
    :return:
                eloggamma :
    """
    eloggamma = psi(eta1) - np.log(eta0)

    return eloggamma


def e_gamma(eta1, eta0):
    """
    :param eta1:
    :param eta0:
    :return:
                eloggamma :
    """
    egamma = eta1/eta0

    return egamma


def c_obs(mu, sigma2, mult=np.array([1.0])):

    c = - 0.5 * (mult[:, None] * (np.square(mu) / sigma2 + np.log(2 * np.pi * sigma2))).sum()

    return np.sum(c)


def calc_entropy(r):
    """
    Compute sum over columns of R * log(R). O(NK) memory. Vectorized.
    Args
    ----
    r : 2D array, N x K
        Each row must have entries that are strictly positive (> 0).
        No bounds checking is enforced!

    Returns
    -------
    H : scalar
        H = np.sum(R[:,:] * log R[:,:])
    """
    eps = np.finfo(float).eps
    h = np.sum(r * np.log(np.clip(r, eps, 1)))
    return h
