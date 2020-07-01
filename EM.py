from __future__ import division, print_function
import numpy as np

from scipy.signal import lfilter, stft
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, correlate2d
import time
import sys

if sys.version_info.major == 3:
    import _pickle as pickle
else:
    import cPickle as pickle




class EM:
    '''
    Class handling the EM algorithm latent variables, parameters and hyper-parameters.

    Initialize with EM(h) where h is a numpy array containing the Room Impulse
    Response (RIR).

    The algorithm is then followed with both the Expectation and Maximization
    steps by the method iteration(). Alternatively, specific steps can be
    computed using the methods E_Step(), M_g(), M_la(), M_a() and M_Step().


    Parameters
    ----------
    h : ndarray of shape (n_samples,)
        Real-valued time-series representation of the RIR.
    P : int, default=20
        Order of the auto-regressive filter g.
    log : TextIOWrapper, default=sys.stdout
        File object used to print any log, default to standart output.


    Attributes
    ----------
    L_h : int
        Length of the input RIR.
    h : ndarray of shape (L_h,)
        Real-valued time-series representation of the input RIR.
    it : int
        Number of past calls of the method iteration().
    P : int
        Order of the auto-regressive filter g.

    mu : ndarray of shape (L_h,)
        Estimated mean of the latent variable b.
    R : ndarray of shape (L_h, L_h)
        Estimated covariance of the latent variable b.

    la : float
        Estimated parameter lambda.
    a : float
        Estimated exponential decrease parameter a.
    sigma2 : float
        Estimated white noise parameter sigma^2.
    A : ndarray of shape (P,)
        AR coefficients of the estimated filter g.

    la_list : list of float
        List of past estimated parameter lambda.
    a_list : list of float
        List of past estimated exponential decrease parameter a.
    sigma2_list : list of float
        List of past estimated white noise parameter sigma^2.
    A_list : list of ndarray of shape (P,)
        List of past AR coefficients of the estimated filter g.
    LP_list : list of float
        List of past log-probabilities computed by the method iteration().
    '''

    def __init__(self, h, P=20, log=sys.stdout):
        np.random.seed(0)
        self.logfile = log
        print("Initializing EM algorithm", file=self.logfile)

        ## Load hyper-parameters and observed variable
        self.h = np.array(h)
        self.L_h = len(h)
        self.d = np.arange(1,self.L_h+1)




        ## Initializing parameters
        self.it = 0
        self.N_first = np.where(np.abs(h) > np.amax(np.abs(h))/2)[0][0] # First spike


        # Initialize la: estimated from N_first (Esp(N_first) = pow(3/la,
        # 1/3)).
        self.la = 3/(np.power(self.N_first,3))

        print("la init = {}".format(self.la), file=self.logfile)


        # Initialize a (decreasing exponential parameter): estimated from the
        # energy of h decreaing exponentially.
        X = np.ones((self.L_h-self.N_first,2))
        X[:,0] = np.arange(self.N_first, self.L_h)
        y = np.log(np.convolve(np.square(h), np.ones(50)/50)+1e-10)[self.N_first:self.L_h]
        self.a = -np.linalg.lstsq(X, y, rcond=None)[0][0]/2

        self.e = np.exp(self.a*np.arange(0,self.L_h))
        self.e2 = np.square(self.e)


        # Initialize A, parameters of the AR filter g: estimated from spectrum
        # of h.
        self.P = P

        f, t, H = stft(h)
        H = np.abs(H)
        autocorr = np.fft.irfft(np.square(np.mean(H[:,:], axis=1)))
        T = toeplitz(autocorr[:self.P],autocorr[:self.P])
        self.alpha_g = np.dot(np.linalg.inv(T), autocorr[1:self.P+1])

        self.A = np.concatenate([[1], -self.alpha_g])

        A_roots = np.roots(self.A)
        A_roots_norm = [r if np.abs(r) < 1 else 1/np.conj(r) for r in A_roots]
        A_poly = np.poly(A_roots_norm)

        self.alpha_g = -A_poly[1:]
        self.A = np.concatenate([[1], -self.alpha_g])
        self.rev_A = self.A[::-1]


        # Initialize sigma2: estimated from values before N_first
        self.sigma2 = np.mean(np.square(self.h[:self.N_first-self.P]))


        ## Initializing latent variables
        self.mu = self.h
        self.R = np.zeros((self.L_h,self.L_h))
        self.E()

        self.LP = 0

        self.LP_list = [self.log_prob()]
        self.A_list = [self.A]
        self.a_list = [self.a]
        self.la_list = [self.la]
        self.sigma2_list = [self.sigma2]


    def _propagate_mu(self):
        """ Propagates any changes made to mu. """
        self.w = self.h-self.mu

        self.mu_pad = np.pad(self.mu, (self.P, 0), 'constant')
        self.M_mu = np.lib.stride_tricks.as_strided(self.mu_pad,
                                               shape=[self.L_h, self.P+1],
                                               strides=[self.mu_pad.strides[-1], self.mu_pad.strides[-1]])
        self.pie = np.dot(self.M_mu, self.rev_A)
        self.pi = self.pie*self.e
        self.p = self.pi*self.d

    def _propagate_R(self):
        """ Propagates any changes made to R. """
        self.R_pad = np.pad(self.R, [(self.P, 0), (0, 0)], 'constant')
        M_R = np.lib.stride_tricks.as_strided(self.R_pad,
                                               shape=[self.L_h, self.L_h, self.P+1],
                                               strides=[self.R_pad.strides[0], self.R_pad.strides[1], self.R_pad.strides[0]])

        self.half_pie_var = np.dot(M_R, self.rev_A)
        self.half_pie_var_pad = np.pad(self.half_pie_var, [(0, 0), (self.P, 0)], 'constant')
        self.M_half_pie_var_pad = np.lib.stride_tricks.as_strided(self.half_pie_var_pad,
                                               shape=[self.L_h, self.P+1],
                                               strides=[self.half_pie_var_pad.strides[0]+self.half_pie_var_pad.strides[1], self.half_pie_var_pad.strides[1]])

        self.pie_var = np.dot(self.M_half_pie_var_pad, self.rev_A)

    def _propagate_A(self):
        """ Propagates any changes made to A. """
        A_roots = np.roots(self.A)
        A_roots_norm = [r if np.abs(r) < 1 else 1/np.conj(r) for r in A_roots]
        A_poly = np.poly(A_roots_norm)
        self.alpha_g = -A_poly[1:]
        self.A = np.concatenate([[1], -self.alpha_g])

        self.rev_A = self.A[::-1]

        self.pie = np.dot(self.M_mu, self.rev_A)
        self.pi = self.pie*self.e
        self.p = self.pi*self.d


        M_R = np.lib.stride_tricks.as_strided(self.R_pad,
                                       shape=[self.L_h, self.L_h, self.P+1],
                                       strides=[self.R_pad.strides[0], self.R_pad.strides[1], self.R_pad.strides[0]])
        self.half_pie_var = np.dot(M_R, self.rev_A)
        self.half_pie_var_pad = np.pad(self.half_pie_var, [(0, 0), (self.P, 0)], 'constant')
        self.M_half_pie_var_pad = np.lib.stride_tricks.as_strided(self.half_pie_var_pad,
                                               shape=[self.L_h, self.P+1],
                                               strides=[self.half_pie_var_pad.strides[0]+self.half_pie_var_pad.strides[1], self.half_pie_var_pad.strides[1]])

        self.pie_var = np.dot(self.M_half_pie_var_pad, self.rev_A)

    def _propagate_a(self):
        """ Propagates any changes made to a. """
        self.e = np.exp(self.a*np.arange(0,self.L_h))
        self.e2 = np.square(self.e)
        self.pi = self.pie*self.e
        self.p = self.pi*self.d


    def copy(self, old):
        """
        Deeply copies the attributes from old, from the same class, to the
        current object.
        """
        self.h = old.h
        self.L_h = old.L_h

        self.d = np.arange(1,self.L_h+1)

        self.it = old.it
        self.N_first = old.N_first
        self.la = old.la
        self.a = old.a
        self.e = np.copy(old.e)
        self.e2 = old.e2

        self.P = old.P
        self.alpha_g = np.copy(old.alpha_g)
        self.A = np.copy(old.A)
        self.sigma2 = old.sigma2
        self.mu = np.copy(old.mu)
        self.R = np.copy(old.R)

        self.b = np.copy(old.mu)
        self.w = np.copy(old.w)
        self.pie = np.copy(old.pie)
        self.pi = np.copy(old.pi)
        self.p = np.copy(old.p)

        self.mu_pad = np.copy(old.mu_pad)
        self.M_mu = np.copy(old.M_mu)
        self.R_pad = np.copy(old.R_pad)
        #self.M_R = np.copy(old.M_R)

        self.half_pie_var = np.copy(old.half_pie_var)
        self.half_pie_var_pad = np.copy(old.half_pie_var_pad)
        self.M_half_pie_var_pad = np.copy(old.M_half_pie_var_pad)
        self.pie_var = np.copy(old.pie_var)

        self.rev_A = np.copy(old.rev_A)

        self.LP = old.LP
        self.LP_list = old.LP_list
        self.la_list = old.la_list
        self.a_list = old.a_list
        self.sigma2_list = old.sigma2_list
        self.A_list = old.A_list


    def log_prob(self):
        """
        Computes and returns the a posteriori expectation of the log-probability.
        """
        res = -self.L_h/2*np.log(2*np.pi*self.la)
        res = res + self.L_h*(self.L_h-1)/2*self.a


        res = res - 1/(2*self.la)*np.square(np.linalg.norm(self.e*self.pie))

        res = res - 1/(2*self.la)*np.sum(self.e2*self.pie_var)

        res = res - self.L_h/2*np.log(2*np.pi*self.sigma2)
        res = res - 1/(2*self.sigma2)*(np.square(np.linalg.norm(self.w))+np.trace(self.R))

        print("Log-probability difference = {}".format(res - self.LP), file=self.logfile)
        self.LP = res
        return res


    def E(self):
        """ Compute the regular, slow version of the Expectation step. """

        print("", file=self.logfile)
        print("Updating R", file=self.logfile)


        TAE = toeplitz(self.A*self.e2[:self.P+1], np.zeros(self.P+1))
        TA = toeplitz(self.A, np.zeros(self.P+1))
        M = np.dot(TAE.transpose(), TA)
        res = toeplitz(np.concatenate([M[:,0], np.zeros((self.L_h-self.P-1))]),
                       np.concatenate([M[0,:], np.zeros((self.L_h-self.P-1))]))
        res[-self.P:, -self.P:] = M[1:,1:]
        res = res*np.array([self.e2]).transpose()
        self.R = self.la*self.sigma2*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*res)



        print("", file=self.logfile)
        print("Updating mu", file=self.logfile)
        self.mu = np.dot(self.R, self.h)/self.sigma2


        # Propagate
        self._propagate_mu()
        self._propagate_R()

    def E_Kalman(self):
        """ Compute the Kalman filter-based version of the Expectation step. """
        print("E step", file=self.logfile)
        # Computes the estimated mu and R using an RTS smoother after a Kalman filtering

        self.P = self.P+1

        self.R[:] = 0
        self.mu[:] = 0

        F = np.eye(self.P, k=-1)
        F[0,:-1] = self.alpha_g
        Q = np.eye(self.P, 1)
        C = Q.transpose()

        # 1 : Forward pass
        mu_prio = np.zeros((self.P, self.L_h))
        R_prio = np.zeros((self.P, self.P, self.L_h))
        mu_post = np.zeros((self.P, self.L_h))
        R_post = np.zeros((self.P, self.P, self.L_h))

        # Init on u = 0
        R_prio[0,0,0] = R_prio[0,0,0]+self.la/self.e2[0]

        # Update
        residual = self.h[0] - mu_prio[0,0]
        residual_cov = R_prio[0,0,0] + self.sigma2
        K = R_prio[:,0,0]/residual_cov

        mu_post[:,0] = mu_prio[:, 0] + K*residual

        K_mat = np.eye(self.P)
        K_mat[:,0] = K_mat[:,0] - K
        R_post[:,:,0] = np.dot(K_mat,R_prio[:,:,0])

        rescovvec = np.zeros(self.L_h)

        for u in range(1,self.L_h):
            # Predict
            mu_prio[1:,u] = mu_post[:-1,u-1]
            mu_prio[0,u] = np.dot(self.alpha_g,mu_post[:-1,u-1])



            R_prio[1:,:,u] = R_post[:-1,:,u-1]
            R_prio[0,:,u] = np.dot(self.alpha_g, R_post[:-1,:,u-1])
            R_prio[:,1:,u] = R_prio[:,:-1,u]
            R_prio[:,0,u] = np.dot(self.alpha_g, R_prio[:,1:,u].transpose())

            R_prio[0,0,u] = R_prio[0,0,u]+self.la/self.e2[u]

            # Update
            residual = self.h[u] - mu_prio[0,u]#np.dot(C, mu_prio[:,u])
            residual_cov = R_prio[0,0,u] + self.sigma2
            rescovvec[u] = residual_cov

            K = R_prio[:,0,u]/residual_cov
            mu_post[:,u] = mu_prio[:, u] + K*residual


            R_post[:,:,u] = R_prio[:,:,u] - np.dot(K[:,np.newaxis], R_prio[0:1,:,u])



        # 2 : Backward pass
        mu_smooth = np.zeros((self.P, self.L_h))
        mu_smooth[:, -1] = mu_post[:, -1]
        self.mu[-1] = mu_smooth[0, -1]

        R_smooth = np.zeros((self.P, self.P, self.L_h))
        R_smooth[:,:,-1] = R_post[:,:,-1]
        self.R[-self.P:,-1] = np.flip(R_smooth[:,0, -1])
        self.R[-1,-self.P:] = np.flip(R_smooth[0,:, -1])


        for u in range(self.L_h-1, self.P-1, -1):
            J = R_post[:,:-1,u-1]
            J = np.concatenate([np.dot(self.alpha_g, J.transpose())[:,np.newaxis],J], axis=1)
            J = np.dot(J, np.linalg.inv(R_prio[:,:,u]))

            mu_smooth[:,u-1] = mu_post[:,u-1] + np.dot(J,mu_smooth[:,u] - mu_prio[:,u])
            self.mu[u-1] = mu_smooth[0, u-1]

            R_smooth[:,:,u-1] = R_post[:,:,u-1] + np.dot(np.dot(J,R_smooth[:,:,u] - R_prio[:,:,u]),J.transpose())


            if u > self.P:
                self.R[u-self.P:u, u-1] = np.flip(R_smooth[:,0,u-1])
                self.R[u-1,u-self.P:u] = np.flip(R_smooth[0,:,u-1])

            if u == self.P:
                self.R[u-self.P:u, u-self.P:u] = np.flip(R_smooth[:,:,u-1])



        self.mu[:self.P] = np.flip(mu_smooth[:,self.P-1])

        self.P = self.P-1

        # Propagate
        self._propagate_mu()
        self._propagate_R()


    def M_g(self):
        """ Computes the maximization step for the parameter g. """

        print("", file=self.logfile)
        print("Updating g", file=self.logfile)
        M_mu1 = np.lib.stride_tricks.as_strided(self.mu_pad,
                                               shape=[self.P+1, self.L_h],
                                               strides=[self.mu_pad.strides[-1], self.mu_pad.strides[-1]])

        M_mu1 = M_mu1[::-1,:]
        M_mu2 = np.transpose(M_mu1[1:,:])
        M_mu1 = M_mu1*self.e2

        M_mu = np.dot(M_mu1, M_mu2)
        v_mu = M_mu[0,:]
        M_mu = M_mu[1:,:]

        M_R = np.zeros((self.P,self.P+1))
        for p in range(1,self.P+1):
            for q in range(0,self.P+1):
                M_R[p-1,q] = np.sum(np.diag(self.R, q-p)[:self.L_h-max(p,q)]*self.e2[max(p,q):self.L_h])

        v_R = M_R[:,0]
        M_R = M_R[:,1:]

        self.alpha_g = np.dot(np.linalg.inv(M_mu + M_R), v_mu+v_R)
        self.A = np.concatenate([[1], -self.alpha_g])

        self._propagate_A()

    def M_la(self):
        """ Computes the maximization step for the parameter lambda. """

        print("", file=self.logfile)
        print("Updating lambda", file=self.logfile)


        self.la = 1/self.L_h*(np.square(np.linalg.norm((self.e*self.pie))))
        self.la = self.la + 1/self.L_h*(np.sum(self.e2*self.pie_var))

    def _F_a(self, x, la_dep=True, calc=True):
        if calc:
            self.vec_F_a = np.square(self.pie) + self.pie_var


        if la_dep:
            return np.sum(self.vec_F_a * ((self.L_h-1)/2 - np.arange(self.L_h)) * np.exp(2*x*np.arange(self.L_h)))
        else:
            return self.L_h*(self.L_h-1)/2 - 1/self.la * np.sum(self.vec_F_a * np.arange(1,self.L_h+1) * np.exp(2*x*np.arange(self.L_h)))

    def M_a(self, la_dep=True):
        """ Computes the maximization step for the parameter a. """

        print("Updating a", file=self.logfile)
        # Initialize vec_F_a
        self.vec_F_a = np.square(self.pie) + self.pie_var

        min = 0
        max = self.a

        while (self._F_a(max, la_dep=la_dep, calc=False) > 0):
            max = 2*max

        for it in range(40):
            if self._F_a((min+max)/2, la_dep=la_dep, calc=False) > 0:
                min = (min+max)/2
            else:
                max = (min+max)/2

        self.a = (min+max)/2

        self._propagate_a()

    def M_sigma(self):
        """ Computes the maximization step for the parameter sigma2. """

        print("", file=self.logfile)
        print("Updating sigma_2", file=self.logfile)
        self.sigma2 = 1/self.L_h*(np.square(np.linalg.norm(self.w)) + np.trace(self.R))


    def iteration(self, E="kalman", la_dep=True):
        """
        Iterates over Expectation and Maximization steps once and stores the
        parameters at each iteration.

        Parameters
        ----------
        E : string, default="kalman"
            Specifies which algorithm to use for the expectation step :
            "kalman" or "slow". Default is "kalman".
        la_dep : bool, default=True
            Specify whether to consider a dependency of a on lambda in the
            maximization with respect to a. Default is True.
        """
        # M step
        self.M_g()

        print("A = {}".format(self.A), file=self.logfile)
        print("", file=self.logfile)

        self.A_list.append(self.A)


        self.M_a(la_dep=la_dep)

        print("a = {}".format(self.a), file=self.logfile)
        print("", file=self.logfile)
        self.a_list.append(self.a)


        self.M_la()


        print("lambda = {}".format(self.la), file=self.logfile)
        print("", file=self.logfile)
        self.la_list.append(self.la)

        self.M_sigma()

        print("sigma_2 = {}".format(self.sigma2), file=self.logfile)
        self.sigma2_list.append(self.sigma2)


        # E step
        if E == "slow":
            self.E()
        elif E == "kalman":
            self.E_Kalman()
        else:
            raise Exception('E should be slow or kalman')

        # LP history
        print("", file=self.logfile)
        self.LP_list.append(self.log_prob())
        print("last free energy : {}".format(self.LP_list[-1]), file=self.logfile)


def save_EM(algo, filename):
    """
    Smartly saves the object of class EM in a given file using pickle.

    Parameters
    ----------
    algo : EM object
        The object to save.
    filename : string
        The path of the file to which save.
    """
    if isinstance(algo, list):
        del algo[0].R
        del algo[0].R_pad
        del algo[0].half_pie_var
        del algo[0].half_pie_var_pad
    elif isinstance(algo, dict):
        del algo["algo"].R
        del algo["algo"].R_pad
        del algo["algo"].half_pie_var
        del algo["algo"].half_pie_var_pad
    else:
        del algo.R
        del algo.R_pad
        del algo.half_pie_var
        del algo.half_pie_var_pad

    with open(filename, 'wb') as output:
        pickle.dump(algo, output, pickle.HIGHEST_PROTOCOL)


def load_EM(filename, E=True):
    """
    Smartly loads the object of class EM in a given file using pickle.

    Parameters
    ----------
    filename : string
        The path of the file to load.
    E : bool, default=True
        Specifies whether to immediatly compute an E step to recompute the R
        matrix. Disable to save time if you are only loading to look at the
        results, not to keep updating.
    """
    with open(filename, 'rb') as input:
        algo = pickle.load(input)

    if E:
        if isinstance(algo, list):
            algo[0].R = np.zeros((algo[0].L_h,algo[0].L_h))
            algo[0].E_Kalman()
        elif isinstance(algo, dict):
            algo["algo"].R = np.zeros((algo["algo"].L_h,algo["algo"].L_h))
            algo["algo"].E_Kalman()
        else:
            algo.R = np.zeros((algo.L_h,algo.L_h))
            algo.E_Kalman()
    return algo


