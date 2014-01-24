from __future__ import division
import numpy as np

EPSILON=1e-7

def frequency_constrained_meanshift(X,k,tau,sigmasq,niter=10):
    """
    Parameters
    ----------
    X :  shape = (n_times, n_freqs)
        Data assumed to be binary in time and frequency
    k : int
        how many quantized frequencies per frequency band in the
        original data for finding means

    tau : int
        time radius for how many adjacent time points should
        be used in the kde
    sigmasq : float
        bandwidth parameter for mean shift
    """
    n_times, n_freqs = X.shape
    nz_times, nz_freqs = np.where(X)
    Y = np.zeros((n_times,n_freqs),dtype=np.uint8)
    for t in xrange(n_times):
        Fs = np.arange(k*n_freqs,dtype=np.float)/k
        Fssq = Fs**2
        use_idx = np.abs(nz_times -t ) <= tau
        use_times = nz_times[use_idx]
        use_freqs = nz_freqs[use_idx]
        time_distances = (t-use_times)**2

        # quit if you can't use any of the times
        if np.sum(use_idx) == 0: continue

        for i in xrange(niter):
            D = np.exp(- ((( (-2)*np.outer(Fs,use_freqs) + Fssq[:,np.newaxis]) + use_freqs**2) + time_distances )/ (2*sigmasq))
            Fs = np.dot(D,use_freqs) / D.sum(1)
            Fssq = Fs**2


        I = np.unique((oned_fixed_points(Fs*k)/k + .5).astype(int))
        Y[t][I] = 1

    return Y


def oned_fixed_points(M):
    M = (M + .5 ).astype(int)
    I = -1* np.ones(len(M),dtype=int)
    S = -1*np.ones(len(M),dtype=int)
    sidx = -1
    for w in xrange(len(M)):
        max_not_found = True
        while max_not_found:
            try:
                equality_test = M[w] == w
            except:
                import pdb; pdb.set_trace()
            if equality_test:
                max_not_found = False
                I[w] = w
                while sidx >= 0:
                    I[S[sidx]] = w
                    sidx = sidx -1
            elif I[w] > -1:
                max_not_found = False
                while sidx >= 0:
                    I[S[sidx]] = w
                    sidx = sidx -1
            else:
                sidx += 1
                S[sidx] = w
                w = M[w]
    return I

def basic_reassignment_fixed_points(t_hat,f_hat,t_q,f_q):
    """
    Parameters
    ----------
    t_hat : (n_times, n_freqs)
        Reassignment operator for times
    f_hat : (n_times, n_freqs)
        Reassignment operator for frequencies
    t_q : int
        Quantization factor for times
    f_q : int
        Quantization factor for frequencies
    """
    n_times, n_freqs = t_hat.shape
    # mapped coordinates for time and frequency
    Mt = -1 * np.ones( (n_times,n_freqs),dtype=int)
    Mf = Mt.copy()
    # stack and index
    Ut = -1 * np.ones( n_times*n_freqs,dtype=int)
    Uf = Ut.copy()
    uidx = -1

    for t in xrange(n_times):
        for f in xrange(n_freqs):
            NotConverged = True
            while NotConverged:
                if Mt[t,f] != -1 and Mf[t,f] != -1:
                    NotConverged = False
                else:
                    uidx +=1
                    Ut[uidx] = t
                    Uf[uidx] = f
                    t1 = t_hat[t,f]
                    f1 = f_hat[t,f]
                    Mt[t,f] = int(t1/t_q + .5)
                    Mf[t,f] = int(f1/f_q + .5)
                    t = t1
                    f = f1

                while uidx >= 0:
                    Mt[Ut[uidx],Uf[Uidx]] = Mt[t,f]
                    Mf[Ut[uidx],Uf[Uidx]] = Mf[t,f]
                    uidx = uidx-1
