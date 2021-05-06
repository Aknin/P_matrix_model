from __future__ import division, print_function
import numpy as np

from scipy.linalg import toeplitz, inv
import scipy.signal as signal
np.set_printoptions(suppress=True)
import time
from numpy.polynomial import polynomial as poly
import sys
import pickle


class P_class:
    '''
    Class handling the matrices P of coefficients P(i,j) = p^{\ast j}(i-j), where p is a filter.

    Initialize with P_class(p).

    Use P.matrix(T) to compute and return the full matrix of shape T x T and P.dot(x) to directly compute Px.
    '''

    def __init__(self, p):
        self.p = (np.array(p))
        self.L_p = self.p.shape[0]

    def matrix(self, T, square=True):
        N_rows = T if square else (T-1)*self.L_p+1
        P = np.zeros((N_rows,T))

        P[0,0] = 1
        P[1:self.L_p+1,1] = self.p
        p_power_i = self.p
        for i in range(2,T):
            p_power_i = np.convolve(p_power_i, self.p)
            P[i:min(i+p_power_i.shape[0], N_rows),i] = p_power_i[:min(p_power_i.shape[0], N_rows-i)]
        return P

    def dot(self, x, test=True):
        T = x.shape[0]
        if test:
            Pm = self.matrix(T)
            return np.dot(Pm, x)
        L = 1+(T-1)*self.L_p

        Pk = np.fft.fft(self.p, L)
        ek = np.exp(-2j*np.pi*np.arange(L)/L)
        zk = Pk*ek

        if x.ndim == 1:
            Yk = poly.Polynomial(x)(zk)
            res = np.real(np.fft.ifft(Yk))[:T]
        else:
            N_col = x.shape[1]
            res = np.zeros((T, N_col))
            for i in range(N_col):
                Yk = poly.Polynomial(x[:,i])(zk)
                res[:,i] = np.real(np.fft.ifft(Yk))[:T]

        return res


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
    L_g : int, default=21
        Order of the auto-regressive filter of coefficients g.
    L_p : int, default=10
        Order of the filter p.
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
    L_g : int
        Order of the auto-regressive filter of coefficients g.
    L_p : int
        Order of the filter p.

    mu : ndarray of shape (L_h,)
        Estimated mean of the latent variable b.
    R : ndarray of shape (L_h, L_h)
        Estimated covariance of the latent variable b.

    la : float
        Estimated parameter lambda.
    a : float
        Estimated exponential decrease parameter a.
    p : ndarray of shape (L_p,)
        Estimated filter used to construct the matrix P.
    sigma2 : float
        Estimated white noise parameter sigma^2.
    g : ndarray of shape (L_g,)
        AR coefficients of the estimated filter G^{-1}.

    la_list : list of float
        List of past estimated parameter lambda.
    a_list : list of float
        List of past estimated exponential decrease parameter a.
    p_list : list of ndarray of shape (L_p,)
        List of past estimated filters used to construct the matrix P.
    sigma2_list : list of float
        List of past estimated white noise parameter sigma^2.
    g_list : list of ndarray of shape (L_g,)
        List of past AR coefficients of the estimated filter G^{-1}.
    LP_list : list of float
        List of past log-probabilities computed by the method iteration().
    '''
    def __init__(self, h, L_g=20, L_p=10, log=sys.stdout, la=None, p=None, g=None, sigma2=None, a=None):
        np.random.seed(0)
        self.logfile = log
        print("Initialization of the EM algorithm", file=self.logfile)

        ## Load hyper-parameters and observed variable
        self.h = np.array(h)
        self.L_h = len(h)

        self.d = np.arange(1, self.L_h+1)
        self.d2 = np.square(self.d)


        ## Initializing parameters
        self.it = 0
        self.N_first = np.where(np.abs(h) > np.amax(np.abs(h))/2)[0][0] # First spike
        self.L_p = L_p
        self.L_g = L_g

        # Initialize la
        if la is None:
            self.la = 3/(np.power(max(self.N_first,1),3))       # Estimated from N_first
        elif la == "rand":
            c = 340
            fs = 16000
            la_i = 20*np.random.randn(1)+100
            self.la = 4*np.pi*(la_i[0])*c**3/fs**3
        else:
            self.la = la

        print("la init = {}".format(self.la), file=self.logfile)


        # Initialize a (decreasing exponential parameter)
        if a is None:
            X = np.ones((self.L_h-self.N_first, 2))
            X[:,0] = np.arange(self.N_first, self.L_h)
            y = np.log(np.convolve(np.square(h), np.ones(50)/50)+1e-10)[self.N_first:self.L_h]
            self.a = -np.linalg.lstsq(X, y, rcond=None)[0][0]/2  #Estimated from the energy of h decreasing exponentially
        elif a == "rand":
            self.a = np.abs(np.random.randn(5)*0.1)
        else:
            self.a = a

        self.e = np.exp(self.a*np.arange(self.L_h))
        self.e2 = np.square(self.e)


        if p is None:
            self.p = np.concatenate([[1],np.zeros(self.L_p-1)])
            self.P = P_class(self.p)
        elif isinstance(p, np.ndarray):
            self.L_p = p.shape[0]
            self.p = np.copy(p)
            self.P = P_class(self.p)
        elif p == "rand":
            self.p = np.concatenate([[1],np.zeros(self.L_p-1)])
            self.p[1:] = self.p[1:] + np.random.randn(self.L_p-1)*0.01
            self.P = P_class(self.p)
        else:
            self.L_p = p.shape[0]
            self.p = np.copy(p)
            self.P = P_class(self.p)


        #Initialize G
        if g is None:
            #Estimated from spectrum of h
            f, t, H = signal.stft(h)
            H = np.abs(H)
            autocorr = np.fft.irfft(np.square(np.mean(H[:,:], axis=1)))
            T = toeplitz(autocorr[:self.L_g-1], autocorr[:self.L_g-1])
            self.g_plus = -np.dot(np.linalg.inv(T), autocorr[1:self.L_g])


            self.g = np.concatenate([[1], self.g_plus])

            g_roots = np.roots(self.g)
            g_roots_norm = [r if np.abs(r)<1 else 1/np.conj(r) for r in g_roots]
            g_poly = np.poly(g_roots_norm)

            self.g_plus = g_poly[1:]
            self.g = np.concatenate([[1], self.g_plus])
            self.rev_g = self.g[::-1]

        elif isinstance(g, np.ndarray):
            self.L_g = g.shape[0]
            self.g_plus = g[1:]
            self.g = np.concatenate([[1], self.g_plus])
            self.rev_g = self.g[::-1]
        elif g == "rand":
            self.g_plus = np.random.randn(20)*0.3
            self.g = np.concatenate([[1], self.g_plus])
            self.rev_g = self.g[::-1]
            print(self.g)
        else:
            self.L_g = g.shape[0]
            self.g_plus = g[1:]
            self.g = np.concatenate([[1], self.g_plus])
            self.rev_g = self.g[::-1]


        # Initialize sigma2
        if sigma2 is None:
            self.sigma2 = np.mean(np.square(self.h[:max(self.N_first-self.L_g+1, int(self.L_g/2))]))   #Estimated from values before N_first
        elif sigma2 == "rand":
            self.sigma2 = np.abs(np.random.randn())*1e-9
            print(self.sigma2)
        else:
            self.sigma2 = sigma2



        # Initialize latent variables
        self.mu = self.h
        self.R = np.zeros((self.L_h, self.L_h))
        self.E()

        self.FE = 0

        self.FE_list = []
        self.free_energy()
        self.g_list = [self.g]
        self.p_list = [self.p]
        self.a_list = [self.a]
        self.la_list = [self.la]
        self.sigma2_list = [self.sigma2]

    def propagate_mu(self):
        self.w = self.h-self.mu


    def propagate_g(self):
        self.g = np.concatenate([[1], self.g_plus])

        self.rev_g = self.g[::-1]



    def free_energy(self):
        res = -self.L_h/2*np.log(2*np.pi*self.la)
        res = res + self.L_h*(self.L_h-1)/2*self.a

        G = toeplitz(np.concatenate([self.g, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))
        t0 = time.time()
        res = res - 1/(2*self.la)*np.square(np.linalg.norm((np.dot(G, self.P.dot(self.e*self.mu)))))


        res = res - 1/(2*self.la)*np.sum(np.diag(np.dot(np.dot(G,self.P.dot(self.P.dot(self.e*self.e[:,np.newaxis]*self.R).transpose()).transpose()),G.transpose())))

        res = res - self.L_h/2*np.log(2*np.pi*self.sigma2)
        res = res - 1/(2*self.sigma2)*(np.square(np.linalg.norm(self.h-self.mu))+np.trace(self.R))

        print("diff = {} (should be positive after M step)".format(res - self.FE), file=self.logfile)
        self.FE = res
        self.FE_list.append(res)
        return res

    def E(self):
        print("", file=self.logfile)
        print("Updating R", file=self.logfile)


        G = toeplitz(np.concatenate([self.g, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))
        GP = np.dot(G, self.P.matrix(self.L_h))


        M = self.e*self.e[:,np.newaxis]*np.dot(GP.transpose(), GP)

        self.R = self.la*self.sigma2*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*M)

        print("", file=self.logfile)
        print("Updating mu", file=self.logfile)
        self.mu = np.dot(self.R, self.h)/self.sigma2


        # Propagate
        self.propagate_mu()

    def M_g(self):
        print("", file=self.logfile)
        print("Updating g", file=self.logfile)
        mat = np.zeros((self.L_g-1,self.L_g-1))
        vec = np.zeros((self.L_g-1,))


        R_tilda = self.R + np.dot(self.mu[:,np.newaxis], self.mu[np.newaxis,])

        PEREP = self.P.dot(self.P.dot(self.e*self.e[:,np.newaxis]*R_tilda).transpose()).transpose()

        traces_plus = np.zeros(self.L_g)
        traces_minus = np.zeros(self.L_g)
        for i in range(self.L_g):
            traces_plus[i] = np.trace(PEREP, offset=i)
            if i>0:
                traces_minus[i] = np.trace(PEREP, offset=-i)
            else:
                traces_minus[i] = traces_plus[i]

        mat = toeplitz(traces_plus, traces_minus)
        vec_plus = mat[1:,0]
        mat_plus = mat[1:,1:]

        cumsum_matrix = np.flip(PEREP[-self.L_g+1:,-self.L_g+1:]).transpose()
        mat_plus = mat_plus - cumsum_matrix
        for i in range(1,self.L_g):
            mat_plus[i:,i:] = mat_plus[i:,i:] - cumsum_matrix[:-i,:-i]



        self.g_plus = np.linalg.solve(mat_plus, -vec_plus)
        self.g = np.concatenate([[1], self.g_plus])


        self.propagate_g()

        self.g_list.append(self.g)

    def M_la(self):
        print("", file=self.logfile)
        print("Updating lambda", file=self.logfile)

        R_tilda = self.R + np.dot(self.mu[:,np.newaxis], self.mu[np.newaxis,])
        PEREP = self.P.dot(self.P.dot(self.e*self.e[:,np.newaxis]*R_tilda).transpose())

        G = toeplitz(np.concatenate([self.g, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))
        tr_GPRPG = np.sum(np.einsum('ij,ji->i', G, np.dot(PEREP, G.transpose())))
        self.la = tr_GPRPG/self.L_h

        self.la_list.append(self.la)

    def M_sigma(self):
        print("", file=self.logfile)
        print("Updating sigma", file=self.logfile)

        self.sigma2 = 1/self.L_h*(np.square(np.linalg.norm(self.w)) + np.trace(self.R))
        self.sigma2_list.append(self.sigma2)

    def M_P(self, p0=False, step = 1e-7):
        print("", file=self.logfile)
        print("Updating P", file=self.logfile)

        G = toeplitz(np.concatenate([self.g, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))
        P_m = self.P.matrix(self.L_h)
        G2 = np.dot(inv(P_m), np.dot(G, P_m))

        grad_p = np.zeros(self.L_p)
        hessian_p = np.eye(self.L_p)

        R_tilda = self.R + np.dot(self.mu[:,np.newaxis], self.mu[np.newaxis,])
        GRG = np.dot(G2,np.dot(R_tilda, G2.transpose()))
        D = np.diag(np.arange(self.L_h))
        P2 = np.zeros((self.L_h, self.L_h))
        P2[1:,1:] = P_m[:-1,:-1]

        mat_grad = np.dot(P2, np.dot(D, np.dot(GRG,P_m.transpose())))

        for i in range(self.L_p):
            grad_p[i] = -1/self.la*np.trace(mat_grad, offset=i)

        grad_p[0] = grad_p[0] + 1/2*self.L_h*(self.L_h-1)


        self.p[(1-p0):] = self.p[(1-p0):] + step*grad_p[(1-p0):]
        self.P = P_class(self.p)

        self.p_list.append(self.p)


    def M_P_newton(self, p0=False, step=1):
        print("", file=self.logfile)
        print("Updating P with Newton's method", file=self.logfile)

        G = toeplitz(np.concatenate([self.g, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))
        P_m = self.P.matrix(self.L_h)
        P_m_inv = inv(P_m)


        G2 = np.dot(P_m_inv, np.dot(G, P_m))

        grad_p = np.zeros(self.L_p)
        hessian_p = np.eye(self.L_p)

        R_tilda = self.R + np.dot(self.mu[:,np.newaxis], self.mu[np.newaxis,])
        GEREG = np.dot(G2,np.dot(self.e*self.e[:,np.newaxis]*R_tilda, G2.transpose()))


        d = np.arange(self.L_h)[:,np.newaxis]
        P2 = np.zeros((self.L_h, self.L_h))
        P2[1:,1:] = P_m[:-1,:-1]

        GEREGP = np.dot(GEREG,P_m.transpose())

        mat_grad = np.dot(P2, (d*GEREGP))

        for i in range(self.L_p):
            grad_p[i] = -1/self.la*np.trace(mat_grad, offset=i)

        grad_p[0] = grad_p[0] + 1/2*self.L_h*(self.L_h-1)


        P3 = np.zeros((self.L_h, self.L_h))
        P3[1:,1:] = P2[:-1,:-1]

        d2 = (np.arange(self.L_h)*np.arange(-1, self.L_h-1))[:,np.newaxis]


        mat_hess_1 = np.dot(P3, (d2*GEREGP))
        mat_hess_2 = np.dot(P2, (d*np.dot(GEREG,(d*P2.transpose()))))


        for i in range(self.L_p):
            for j in range(self.L_p):
                hessian_p[i,j] = -1/self.la*np.trace(mat_hess_1, offset=i+j)
                hessian_p[i,j] = hessian_p[i,j] -1/self.la*np.trace(mat_hess_2[:-max(i,j),:-max(i,j)], offset=i-j)



        self.p[(1-p0):] = self.p[(1-p0):] - step*np.dot(inv(hessian_p[(1-p0):,(1-p0):]),grad_p[(1-p0):])
        self.P = P_class(self.p)

        self.p_list.append(self.p)



    def F_a(self, x, vec):
        #res = self.L_h*(self.L_h-1)/2 - 1/self.la * np.sum(vec * np.arange(self.L_h) * np.exp(2*x*np.arange(self.L_h)))
        res = np.sum(vec * ((self.L_h-1)/2 - np.arange(self.L_h)) * np.exp(2*x*np.arange(self.L_h)))

        return res

    def M_a(self):
        print("", file=self.logfile)
        print("Updating a", file=self.logfile)
        p_tilda = self.p/self.e[:self.L_p]
        P_tilda = P_class(p_tilda)
        P_tilda_m = P_tilda.matrix(self.L_h)

        g_tilda = self.g/self.e[:self.L_g]
        G_tilda = toeplitz(np.concatenate([g_tilda, np.zeros((self.L_h-self.L_g))]),np.zeros(self.L_h))

        R_tilda = self.R + np.dot(self.mu[:,np.newaxis], self.mu[np.newaxis,])

        GP = np.dot(G_tilda, P_tilda_m)
        GPRPG_diag = np.sum(GP*(np.dot(GP,R_tilda.transpose())), axis=1)


        min = 0
        max = self.a

        while (self.F_a(max, GPRPG_diag) > 0):
            min = max
            max = 2*max

        for it in range(40):
            if self.F_a((min+max)/2, GPRPG_diag) > 0:
                min = (min+max)/2
            else:
                max = (min+max)/2

        self.a = (min+max)/2
        self.e = np.exp(self.a*np.arange(self.L_h))
        self.e2 = np.square(self.e)

        self.a_list.append(self.a)


    def iteration(self, compute_FE=True):
        print("Iteration {}".format(self.it), file=self.logfile)
        self.M_g()
        self.M_sigma()
        self.M_a()
        self.M_la()
        self.M_P_newton(False,step=1)
        self.E()
        print("", flush=True, file=self.logfile)

        if compute_FE:
            self.free_energy()

        self.it = self.it + 1



def save_algo(filename, algo, fs=None, V_room_true=None):
    '''
    Function saving the attributes of a object of the class EM in a pickle file.


    Parameters
    ----------
    filename : string
        Path to the file to save to.
    algo : object of class EM
        Object of the class EM to save.
    fs : int, default=None
        Sampling frequency to save along the EM object.
    V_room_true : float, default=None
        Volume of the room to save along the EM object.
    '''

    dic = {}
    dic["h"] = algo.h
    dic["sigma2"] = algo.sigma2
    dic["lambda"] = algo.la
    dic["g"] = algo.g
    dic["p"] = algo.p
    dic["a"] = algo.a
    dic["FE_list"] = algo.FE_list
    dic["fs"] = fs
    dic["V_room_true"] = V_room_true

    file = open(filename, "wb")
    pickle.dump(dic, file)
    file.close()



def load_algo(filename):
    '''
    Function loading an object of the class EM from a pickle file.

    Parameters
    ----------
    filename : string
        Path to the file to load from.


    Returns
    ----------
    algo : object of class EM
        Object of the class EM loaded from the file.
    fs : int, default=None
        Sampling frequency loaded along the EM object.
    '''

    file = open(filename, "rb")
    dic = pickle.load(file)
    file.close()

    print("Last log-probability evaluation: {}".format(dic["FE_list"][-1]))

    print("Room volume: {}".format(dic["V_room_true"]))


    return EM(dic["h"], sigma2=dic["sigma2"], la=dic["lambda"], g=dic["g"], p=dic["p"], a=dic["a"]), dic["fs"]
