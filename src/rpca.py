import numpy as np
from scipy import linalg as la


class RPCA(object):
    """A Python implementation of Robust PCA algorithm.

    This implementation solves the principle component pursuit (PCP) problem by
    alternating directions. The theory and implementation of the algorithm is
    described here: http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf

    """

    def __init__(self, M, mu=None, lmbda=None, verbose=True):
        self.M = M
        self.verbose = verbose

        if mu is not None:
            self.mu = mu
        else:
            self.mu = np.prod(M.shape) / (4.0 * la.norm(M, ord=1))

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.lmbda = 1.0 / np.sqrt(np.max(M.shape))

        if verbose:
            print 'mu    = ', self.mu
            print 'lmbda = ', self.lmbda

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)

    @staticmethod
    def svd_threshold(M, tau):
        U, s, VT = la.svd(M, full_matrices=False)
        return np.dot(U*RPCA.shrink(s, tau), VT)

    def fit(self, tol=1.0e-7, max_iter=1000, iter_print=100, return_err=False):

        cnt = 0
        mu_inv = 1.0 / self.mu
        Sk = np.zeros_like(self.M)
        Yk = np.zeros_like(self.M)

        MF = la.norm(self.M, ord='fro')

        while cnt < max_iter:
            mYk = mu_inv*Yk
            Lk = self.svd_threshold(self.M - Sk + mYk, mu_inv)
            Sk = self.shrink(self.M - Lk + mYk, self.lmbda*mu_inv)

            err = la.norm(self.M - Lk - Sk, ord='fro') / MF
            if self.verbose and cnt % iter_print == 0:
                print 'Iteration {0}, error = {1}'.format(cnt, err)
            if err < tol:
                if self.verbose and cnt % iter_print != 0:
                    print 'Iteration {0}, error = {1}'.format(cnt, err)
                break
            else:
                Yk = Yk + self.mu * (self.M - Lk - Sk)
                cnt += 1
        else:
            if self.verbose:
                print 'Warn: Exit with max_iter = {0}, error = {1}, tol = {2}'.format(cnt, err, tol)

        if return_err:
            return Lk, Sk, err
        else:
            return Lk, Sk



class MRPCA(RPCA):

    @staticmethod
    def svd_threshold(M, tau):
        s, U = la.eigh(M)
        return np.dot(U*MRPCA.shrink(s, tau), U.T)