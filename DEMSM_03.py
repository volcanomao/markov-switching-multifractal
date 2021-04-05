import numpy as np
import cupy as cp
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from numba import jit, njit, prange, float64, int64

"""Data Enabled Markov-Switching Multifractal
Author: John Collins, john.collins@edhec.com
Version: 0.3, 11-March-2020
Notes: R-squared and MSE added to pred function.
       Sigma is a positive constant (Calvet & Fisher, 2004), but MATLAB code of C&F (2013) adjusts: std(data)*sqrt(252)
       I make no sqrt(time) adjustment, for consistency across datasets. Therefore, sigma = np.std(data)
"""

def glo_min(kbar, data, niter, temperature, stepsize):
    """2-step basin-hopping method combines global stepping algorithm with local minimization at each step.
    Step 1: local minimizations; parameter estimations using bounded optimization (scipy.optimize.fminbound).
    Step 2: global minimum search using basin-hopping algorithm (scipy.optimize.basinhopping).
    """

    """Step 1: local minimizations
    """
    theta, theta_LLs, theta_out, ierr, numfunc = loc_min(kbar, data)

    """Step 2: global minimum search using basin-hopping
    """
    # func = Function to be optimized
    f = g_LLb_h

    # x0 = initial guess, being theta, from Step 1.  Presents as: [b, m0, gamma_kbar, sigma]
    x0 = theta

    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                           minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message

    return(parameters, LL, niter, output)


def loc_min(kbar, data):
    """Function for step 1: local minimizations.
    Parameter estimation using bounded optimization (scipy.optimize.fminbound)

    CRITICAL: MAKE SIGMA AN ARGUMENT | to deal with time-variance
    """

    # set up
    b = np.array([1.5, 5, 15, 30])
    lb = len(b)
    gamma_kbar = np.array([0.1, 0.5, 0.9, 0.95])
    lg = len(gamma_kbar)
    sigma = np.std(data)

    # templates
    theta_out = np.zeros(((lb*lg),3))
    theta_LLs = np.zeros((lb*lg))

    # objective function
    f = g_LL

    # bounds
    m0_l = 1.2
    m0_u = 1.8

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence too; 3, iteration results.
    disp = 1

    idx = 0
    for i in range(lb):
        for j in range(lg):

            # args
            theta_in = [b[i], gamma_kbar[j], sigma]
            args = (kbar, data, theta_in)

            xopt, fval, ierr, numfunc = opt.fminbound(func = f, x1 = m0_l, x2 = m0_u, xtol = xtol,
                                               args = args, full_output = True, disp = disp)

            m0, LL = xopt, fval
            theta_out[idx,:] = b[i], m0, gamma_kbar[j]

            theta_LLs[idx] = LL
            idx +=1

    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)

    theta = theta_out[idx[0],:].tolist()+[sigma]
    theta_out = theta_out[idx,:]

    return(theta, theta_LLs, theta_out, ierr, numfunc)


def g_LLb_h(theta, kbar, data):
    """bridges global minimization to local minimization
    """

    theta_in = unpack(theta)
    m0 = theta[1]
    LL = g_LL(m0, kbar, data, theta_in)

    return(LL)


def g_LL(m0, kbar, data, theta_in):
    """[DESCRIBE]
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = sigma*g_m

    # returns
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16

    # log likelihood using numba
    LL = _LL(kbar2, T, A, g_m, w_t)

    return(LL)


@jit(nopython=True)
def _LL(kbar2, T, A, g_m, w_t):
    """[DESCRIBE]
    """

    LLs = np.zeros(T)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)

        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

        # vector of log likelihoods
        LLs[t] = np.log(np.dot(w_t[t,:],piA))

    LL = -np.sum(LLs)

    return(LL)


def g_pi_t(m0, kbar, data, theta_in):
    """returns pi_t, the current distribution of states for prediction
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = sigma*g_m

    # returns
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16

    # compute pi_t with numba acceleration
    pi_t = _t(kbar2, T, A, g_m, w_t)

    return(pi_t)

@jit(nopython=True)
def _t(kbar2, T, A, g_m, w_t):
    """[DESCRIBE]
    """

    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)
        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

    pi_t = pi_mat[-1,:]

    return(pi_t)


def pred(theta, kbar, data, K):
    """Compute K-step ahead forecasts, MZ regressions, MSE and R-squared coefficient
    """

    # set up
    b = theta[0]
    m0 = theta[1]
    gamma_kbar = theta[2]
    sigma = theta[3]
    kbar2 = 2**kbar
    pred_means = np.zeros(K)
    preds = []
    vol_list = []
    y_test = data[-K:]

    # distribution of states for prediction
    theta_in = unpack(theta)
    pi_t = g_pi_t(m0, kbar, data, theta_in)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities [CONFIRM THIS IS WHAT THIS IS]
    g_m = s_p(kbar, m0)

    # K-step vol prediction, MZ regression, MSE and R-squared
    pi_n = np.zeros((K,pi_t.shape[0]))
    for i in range(K):
        A_n = np.array(np.matrix(A)**(i+1))
        pi_n[i,:] = np.dot(pi_t,A_n)
        pred = (sigma)**2*(g_m)
        pred_means[i] = np.average(pred, weights = pi_n[i,:])

    preds.append(pred_means)
    vol_list.append(y_test**2)                        # WHY SQUARED RETURNS?
    X_sim = np.concatenate(preds).ravel()
    X_sim = np.c_[np.ones(len(X_sim)),X_sim]
    Y_sim = np.concatenate(vol_list).ravel()

    # MZ regressions of squared returns on forecasts  (X=pred, Y=true)   # WHY SQUARED RETURNS?
    Y_mz = pd.DataFrame(Y_sim)
    X_mz = pd.DataFrame(X_sim[:,1])
    compare_df = pd.concat([Y_mz, X_mz], axis=1)
    compare_df.columns = ['Y_mz', 'X_mz']
    mz_reg = f'Y_mz ~ X_mz'
    mz_result = sm.ols(mz_reg, data=compare_df).fit()
    g_0 = mz_result.params[0]
    g_1 = mz_result.params[1]

    # Stats: R-square, MSE, F-test, F-test p-value
    r_2 = mz_result.rsquared
    mse = np.mean((compare_df['Y_mz'] - compare_df['X_mz']) ** 2)
    f_test =  mz_result.fvalue
    f_pval = mz_result.f_pvalue

    return(X_sim, Y_sim, g_0, g_1, mse, r_2, f_test, f_pval)


class memoize(dict):
    """I use a memoize decorator to accelerate compute of the transition probability matrix A
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

@memoize
def  g_t(kbar, b, gamma_kbar):
    """I use a memoize decorator to accelerate compute of the transition probability matrix A
    """

    # compute gammas
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]

    # transition probabilities
    kbar2 = 2**kbar
    prob = np.ones(kbar2)

    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]

    A = np.fromfunction(lambda i,j: prob[np.bitwise_xor(i,j)],(kbar2,kbar2),dtype = np.uint16)

    return(A)


# @jit(nopython=True)
# @njit(parallel=True)
# def g_t_jit(kbar, b, gamma_kbar):
    """Vectorized and numba accelerated version of g_t.
    I vectorize the compute of the transition probability matrix A and accelerate it with numba
    """

    # For working version of code, see | DEMSMv2.ipynb

    # return(A)


def j_b(x, num_bits):
    """I vectorize the first part of computing of the transition probability matrix A
    """

    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])

    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


@jit(nopython=True)
def s_p(kbar, m0):
    """[DESCRIBE]
    """

    # switching probabilities
    m1 = 2-m0
    kbar2 = 2**kbar
    g_m = np.zeros(kbar2)
    g_m1 = np.arange(kbar2)

    for i in range(kbar2):
        g = 1
        for j in range(kbar):
            if np.bitwise_and(g_m1[i],(2**j))!=0:
                g = g*m1
            else:
                g = g*m0
        g_m[i] = g

    return(np.sqrt(g_m))


def unpack(theta):
    """unpack theta, package theta_in
    """
    b = theta[0]
    m0 = theta[1]
    gamma_kbar = theta[2]
    sigma = theta[3]

    theta_in = [b, gamma_kbar, sigma]

    return(theta_in)
