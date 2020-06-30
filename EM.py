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
    Class handling the EM algorithm latent variables, parameters and hyper_parameters.

    Initialize with EM(h).

    The algorithm is then followed with the methods E_Step() and M_Step().
    '''

    def __init__(self, h, log=sys.stdout):
        self.logfile = log
        print("Initializing EM algorithm", file=self.logfile)

        ## Load hyper-parameters and observed variable
        self.h = np.array(h)
        self.L_h = len(h)

        self.d = np.arange(1,self.L_h+1)
        self.d2 = np.square(self.d)




        ## Initializing parameters
        self.it = 0
        self.N_first = np.where(np.abs(h) > np.amax(np.abs(h))/2)[0][0] # First spike


        # Initialize la
        #c = 340
        #fs = 16000
        #la_i = 20*np.random.randn(1)+100
        #self.la = 4*np.pi*(la_i[0])*c**3/fs**3         # Specific value
        #self.la = la_u_true                            # True value
        self.la = 3/(np.power(self.N_first,3))         # Estimated from N_first (Esp(N_first) = pow(3/la, 1/3))
                                                                                 # well not really
        print("la init = {}".format(self.la), file=self.logfile)


        # Initialize a (decreasing exponential parameter)
        #T60 = 0.360
        #self.a = 3*np.log(10)/(T60*fs)                         # Specific value
        X = np.ones((self.L_h-self.N_first,2))
        X[:,0] = np.arange(self.N_first, self.L_h)
        y = np.log(np.convolve(np.square(h), np.ones(50)/50)+1e-10)[self.N_first:self.L_h]
        self.a = -np.linalg.lstsq(X, y, rcond=None)[0][0]/2    # Estimated from the energy of h decreasing exponentially
        #self.a = a_true                                        # True value

        self.e = np.exp(self.a*np.arange(0,self.L_h))
        self.e2 = np.square(self.e)


        # Initialize A
        self.P = 20

        #                                                 # Estimated from values between N_first and N_first + P + 1
        #self.delta = np.zeros(self.P+1)
        #self.delta[0] = 1
        #test1 = np.exp(-self.a*np.arange(self.N_first,self.N_first+self.P+1))*1/np.arange(self.N_first,self.N_first+self.P+1)*self.delta
        #test = lfilter([1], h[self.N_first:self.N_first+self.P+1], test1)
        #plt.plot(lfilter([1], test, delta))
        #plt.show()
        #test_roots = np.roots(test)
        #test_roots_norm = [r if np.abs(r) < 1 else 1/np.conj(r) for r in test_roots]
        #poly = np.poly(test_roots_norm)
        #self.alpha_g = -poly[1:]
                                                         # Estimated from spectrum of h
        f, t, H = stft(h)
        H = np.abs(H)
        autocorr = np.fft.irfft(np.square(np.mean(H[:,:], axis=1)))
        T = toeplitz(autocorr[:self.P],autocorr[:self.P])
        self.alpha_g = np.dot(np.linalg.inv(T), autocorr[1:self.P+1])
        self.alpha_g = 20*np.random.randn(self.P)        # Specific value
        #self.alpha_g = -alpha_g_true[1:]                 # True value

        self.A = np.concatenate([[1], -self.alpha_g])

        A_roots = np.roots(self.A)
        A_roots_norm = [r if np.abs(r) < 1 else 1/np.conj(r) for r in A_roots]
        A_poly = np.poly(A_roots_norm)

        self.alpha_g = -A_poly[1:]
        self.A = np.concatenate([[1], -self.alpha_g])
        self.rev_A = self.A[::-1]



        # Initialize sigma2
        #self.sigma2 = 8*1e-9                                           # Specific value
        self.sigma2 = np.mean(np.square(self.h[:self.N_first-self.P])) # Estimated from values before N_first
        #self.sigma2 = sigma2_true                                      # True value



        ## Initializing latent variables
        self.mu = self.h
        self.R = np.zeros((self.L_h,self.L_h))
        self.E()

        self.FE = 0

        self.FE_list = [self.free_energy()]
        self.A_list = [self.A]
        self.a_list = [self.a]
        self.la_list = [self.la]
        self.sigma2_list = [self.sigma2]


    def propagate_mu(self):
        self.w = self.h-self.mu

        self.mu_pad = np.pad(self.mu, (self.P, 0), 'constant')
        self.M_mu = np.lib.stride_tricks.as_strided(self.mu_pad,
                                               shape=[self.L_h, self.P+1],
                                               strides=[self.mu_pad.strides[-1], self.mu_pad.strides[-1]])
        self.pie = np.dot(self.M_mu, self.rev_A)
        self.pi = self.pie*self.e
        self.p = self.pi*self.d

    def propagate_R(self):
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

    def propagate_A(self):
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

    def propagate_a(self):
        self.e = np.exp(self.a*np.arange(0,self.L_h))
        self.e2 = np.square(self.e)
        self.pi = self.pie*self.e
        self.p = self.pi*self.d


    def copy(self, old):
        self.h = old.h
        self.L_h = old.L_h

        self.d = np.arange(1,self.L_h+1)
        self.d2 = np.square(self.d)

        self.it = old.it
        self.N_first = old.N_first # First spike
        self.la = old.la
        self.a = old.a
        self.e = np.copy(old.e)
        self.e2 = old.e2

        # Initialize A
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

        self.FE = old.FE
        self.FE_list = old.FE_list
        self.la_list = old.la_list
        self.a_list = old.a_list
        self.sigma2_list = old.sigma2_list
        self.A_list = old.A_list


    def free_energy_slow(self):
        res = -self.L_h/2*np.log(2*np.pi*self.la)
        res = res + self.L_h*(self.L_h-1)/2*self.a

        TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        t0 = time.time()
        res = res - 1/(2*self.la)*np.square(np.linalg.norm((self.e*np.dot(TA, self.mu))))
        res = res - 1/(2*self.la)*np.sum(self.e2*np.diag(np.dot(np.dot(TA,self.R),TA.transpose())))
        print(time.time() - t0, file=self.logfile)

        res = res - self.L_h/2*np.log(2*np.pi*self.sigma2)
        res = res - 1/(2*self.sigma2)*(np.square(np.linalg.norm(self.h-self.mu))+np.trace(self.R))

        print("diff = {}".format(res - self.FE), file=self.logfile)
        self.FE = res
        return res

    def free_energy(self):
        res = -self.L_h/2*np.log(2*np.pi*self.la)
        res = res + self.L_h*(self.L_h-1)/2*self.a


        #TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        t0 = time.time()
        res = res - 1/(2*self.la)*np.square(np.linalg.norm(self.e*self.pie))


        res = res - 1/(2*self.la)*np.sum(self.e2*self.pie_var)

        #print("time free_energy : {} s".format(time.time() - t0), file=self.logfile)


        res = res - self.L_h/2*np.log(2*np.pi*self.sigma2)
        res = res - 1/(2*self.sigma2)*(np.square(np.linalg.norm(self.w))+np.trace(self.R))

        print("diff = {}".format(res - self.FE), file=self.logfile)
        self.FE = res
        return res


    def E(self):

        print("", file=self.logfile)
        print("Updating R", file=self.logfile)


        #TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        #self.R = self.la*self.sigma2*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))


        TAE = toeplitz(self.A*self.e2[:self.P+1], np.zeros(self.P+1))
        TA = toeplitz(self.A, np.zeros(self.P+1))
        M = np.dot(TAE.transpose(), TA)
        res = toeplitz(np.concatenate([M[:,0], np.zeros((self.L_h-self.P-1))]),
                       np.concatenate([M[0,:], np.zeros((self.L_h-self.P-1))]))
        res[-self.P:, -self.P:] = M[1:,1:]
        res = res*np.array([self.e2]).transpose()
        self.R = self.la*self.sigma2*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*res)

        #print(self.free_energy(), file=self.logfile)


        print("", file=self.logfile)
        print("Updating mu", file=self.logfile)
        self.mu = np.dot(self.R, self.h)/self.sigma2
        #self.R = np.linalg.inv(1/self.sigma2*np.eye(self.L_h) + 1/self.la*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))



        # Propagate
        self.propagate_mu()
        self.propagate_R()

    def E_Kalman(self):
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
        residual = self.h[0] - mu_prio[0,0] #np.dot(C, mu_prio[:,0])
        #residual_cov = np.dot(np.dot(C, R_prio[:,:,0]), C.transpose()) + self.sigma2
        residual_cov = R_prio[0,0,0] + self.sigma2
        #K = np.dot(R_prio[:,:,0],C.transpose())/residual_cov
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



            #R_prio[:,:,u] = np.dot(np.dot(F, R_post[:,:,u-1]), F.transpose())
            R_prio[1:,:,u] = R_post[:-1,:,u-1]
            R_prio[0,:,u] = np.dot(self.alpha_g, R_post[:-1,:,u-1])
            R_prio[:,1:,u] = R_prio[:,:-1,u]
            R_prio[:,0,u] = np.dot(self.alpha_g, R_prio[:,1:,u].transpose())

            R_prio[0,0,u] = R_prio[0,0,u]+self.la/self.e2[u]
            #R_prio[:,:,u] = R_prio[:,:,u] + self.la*np.exp(-2*self.a*np.arange(u,u-self.P,-1))

            # Update
            residual = self.h[u] - mu_prio[0,u]#np.dot(C, mu_prio[:,u])
            residual_cov = R_prio[0,0,u] + self.sigma2
            #residual_cov = np.dot(np.dot(C, R_prio[:,:,u]), C.transpose()) + self.sigma2
            rescovvec[u] = residual_cov

            #if (u < 22):
                #print("u : {}".format(u))
                #print("mu_prio : {}".format(mu_prio[:,u]))
                #plt.imshow(R_prio[:,:,u])
                #plt.colorbar()
                #plt.show()
            #print("res : {}".format(residual))
            #print("res_cov : {}".format(residual_cov))
            K = R_prio[:,0,u]/residual_cov
            #K_mat = np.eye(self.P)
            #K_mat[:,0] = K_mat[:,0] - K
            #K = np.dot(R_prio[:,:,u],C.transpose())/residual_cov

            mu_post[:,u] = mu_prio[:, u] + K*residual


            R_post[:,:,u] = R_prio[:,:,u] - np.dot(K[:,np.newaxis], R_prio[0:1,:,u])
            #R_post[:,:,u] = np.dot(K_mat,R_prio[:,:,u])
            #R_post[:,:,u] = np.dot((np.eye(self.P) - np.dot(K, C)),R_prio[:,:,u])
            #if (u < 22):
                #print("mu_post : {}".format(mu_post[:,u]))
                #plt.imshow(R_post[:,:,u])
                #plt.colorbar()
                #plt.show()
                #print("")



        # 2 : Backward pass
        mu_smooth = np.zeros((self.P, self.L_h))
        mu_smooth[:, -1] = mu_post[:, -1]
        self.mu[-1] = mu_smooth[0, -1]

        R_smooth = np.zeros((self.P, self.P, self.L_h))
        R_smooth[:,:,-1] = R_post[:,:,-1]
        self.R[-self.P:,-1] = np.flip(R_smooth[:,0, -1])
        self.R[-1,-self.P:] = np.flip(R_smooth[0,:, -1])
        #self.R[-self.P:, -self.P:] = np.flip(R_smooth[:,:, -1])


        for u in range(self.L_h-1, self.P-1, -1):
            #print("u : {}".format(u))
            J = R_post[:,:-1,u-1]
            J = np.concatenate([np.dot(self.alpha_g, J.transpose())[:,np.newaxis],J], axis=1)
            J = np.dot(J, np.linalg.inv(R_prio[:,:,u]))
            #J = np.dot(np.dot(R_post[:,:,u-1], F.transpose()), np.linalg.inv(R_prio[:,:,u]))

            mu_smooth[:,u-1] = mu_post[:,u-1] + np.dot(J,mu_smooth[:,u] - mu_prio[:,u])
            self.mu[u-1] = mu_smooth[0, u-1]

            R_smooth[:,:,u-1] = R_post[:,:,u-1] + np.dot(np.dot(J,R_smooth[:,:,u] - R_prio[:,:,u]),J.transpose())
            #self.R[u-1:,u-1] = R_smooth[:min(self.P, self.L_h-u+1),0,u-1]


            if u > self.P:
                #self.R[max(0, u-self.P):(u), max(0, u-self.P):(u)] = np.flip(R_smooth[self.P - min(self.P, u):,self.P - min(self.P, u):, u-1])


                #self.R[u-self.P:u, u-self.P:u] = np.flip(R_smooth[:,:,u-1])
                self.R[u-self.P:u, u-1] = np.flip(R_smooth[:,0,u-1])
                self.R[u-1,u-self.P:u] = np.flip(R_smooth[0,:,u-1])

            if u == self.P:
                self.R[u-self.P:u, u-self.P:u] = np.flip(R_smooth[:,:,u-1])

            #self.R[u-1,u-1:] = R_smooth[:min(self.P, self.L_h-u+1),0,u-1]
            #if u == 21:
                #plt.imshow(np.flip(R_smooth[:,:,u-1]))
                #plt.colorbar()
                #plt.show()


        self.mu[:self.P] = np.flip(mu_smooth[:,self.P-1])

        #plt.plot(self.mu)
        #plt.show()

        self.P = self.P-1

        #plt.imshow(self.R)
        #plt.colorbar()
        #plt.show()

        # Propagate
        self.propagate_mu()
        self.propagate_R()

    def E_slow(self):

        print("", file=self.logfile)
        print("Updating R", file=self.logfile)


        TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        self.R = self.la*self.sigma2*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))
        print(self.free_energy(), file=self.logfile)



        print("", file=self.logfile)
        print("Updating mu", file=self.logfile)
        self.mu = np.dot(self.R, self.h)/self.sigma2
        #self.R = np.linalg.inv(1/self.sigma2*np.eye(self.L_h) + 1/self.la*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))



        # Propagate
        self.propagate_mu()
        self.propagate_R()

    def E_real(self): # Update by maximizing Free_energy wrt R : not working

        print("", file=self.logfile)
        print("Updating R", file=self.logfile)
        TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        self.R = self.la*self.la*np.linalg.inv(self.la*np.eye(self.L_h) + self.sigma2*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))
        self.R = self.R - self.la*self.la/2*np.linalg.inv(np.dot(algo.h[:,np.newaxis], algo.h[np.newaxis,:]))
        print(self.free_energy_slow(), file=self.logfile)


        print("", file=self.logfile)
        print("Updating mu", file=self.logfile)
        self.mu = np.dot(self.R, self.h)/self.sigma2
        #self.R = np.linalg.inv(1/self.sigma2*np.eye(self.L_h) + 1/self.la*np.dot(np.dot(TA.transpose(),np.diag(self.e2)),TA))



        # Propagate
        self.propagate_mu()
        self.propagate_R()


    def M_g_slow(self):

        print("", file=self.logfile)
        print("Updating g (slow version)", file=self.logfile)
        mat = np.zeros((self.P,self.P))
        vec = np.zeros((self.P,))

        R_mat = np.zeros((self.P,self.P))
        R_vec = np.zeros((self.P,))

        for p in range(self.P):
            for u in range(p,self.L_h):
                vec[p] = vec[p] + (self.mu[u]*self.mu[u-p-1] + self.R[u,u-p-1])*np.exp(2*self.a*u)
                R_vec[p] = R_vec[p] + self.R[u,u-p-1]*np.exp(2*self.a*u)
                for q in range(0, min(u , self.P)):
                    R_mat[p,q] = R_mat[p,q] + self.R[u-q-1,u-p-1]*np.exp(2*self.a*u)
                    mat[p,q] = mat[p,q] + (self.mu[u-q-1]*self.mu[u-p-1] + self.R[u-q-1,u-p-1])*np.exp(2*self.a*u)

        self.alpha_g = np.linalg.solve(mat, vec)
        self.A = np.concatenate([[1], -self.alpha_g])


        self.propagate_A()

    def M_g(self):

        print("", file=self.logfile)
        print("Updating g", file=self.logfile)
        # (M_mu1*exp*M_mu1.T + M_R)*alpha = v_mu + v_R
        #mu_pad = np.pad(self.mu, (self.P, 0), 'constant')
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

        self.propagate_A()

    def M_la(self):

        print("", file=self.logfile)
        print("Updating lambda", file=self.logfile)
        #TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        #self.la = 1/self.L_h*(np.square(np.linalg.norm((self.e*np.dot(TA, self.mu)))) +\
        #                      np.sum(self.e2*np.diag(np.dot(np.dot(TA,self.R),TA.transpose()))))


        self.la = 1/self.L_h*(np.square(np.linalg.norm((self.e*self.pie))))
        self.la = self.la + 1/self.L_h*(np.sum(self.e2*self.pie_var))

    def M_la_slow(self):

        print("", file=self.logfile)
        print("Updating lambda", file=self.logfile)
        TA = toeplitz(np.concatenate([self.A, np.zeros((self.L_h-self.P-1))]),np.zeros(self.L_h))
        self.la = 1/self.L_h*(np.square(np.linalg.norm((self.e*np.dot(TA, self.mu)))) +\
                              np.sum(self.e2*np.diag(np.dot(np.dot(TA,self.R),TA.transpose()))))

    def F_a(self, x, la_dep=True, calc=True):
        if calc:
            self.vec_F_a = np.square(self.pie) + self.pie_var


        if la_dep:
            return np.sum(self.vec_F_a * ((self.L_h-1)/2 - np.arange(self.L_h)) * np.exp(2*x*np.arange(self.L_h)))
        else:
            return self.L_h*(self.L_h-1)/2 - 1/self.la * np.sum(self.vec_F_a * np.arange(1,self.L_h+1) * np.exp(2*x*np.arange(self.L_h)))

    def M_a(self, la_dep=True):
        print("Updating a", file=self.logfile)
        # Initialize vec_F_a
        self.vec_F_a = np.square(self.pie) + self.pie_var

        min = 0
        max = self.a

        while (self.F_a(max, la_dep=la_dep, calc=False) > 0):
            max = 2*max

        for it in range(40):
            if self.F_a((min+max)/2, la_dep=la_dep, calc=False) > 0:
                min = (min+max)/2
            else:
                max = (min+max)/2

        self.a = (min+max)/2

        self.propagate_a()

    def M_sigma(self):

        print("", file=self.logfile)
        print("Updating sigma_2", file=self.logfile)
        self.sigma2 = 1/self.L_h*(np.square(np.linalg.norm(self.w)) + np.trace(self.R))


    def iteration(self, E="kalman", la_dep=False):
        #self.R = np.maximum(self.R, 0)
        # M step
        test_roots = np.roots(self.A)
        print("max root before = {}".format(np.amax(np.abs(test_roots))), file=self.logfile)

        t0 = time.time()
        self.M_g()
        print("time M_g : {} s".format(time.time()-t0), file=self.logfile)

        print("A = {}".format(self.A), file=self.logfile)
        print("", file=self.logfile)
        test_roots = np.roots(self.A)
        print("max root = {}".format(np.amax(np.abs(test_roots))), file=self.logfile)

        self.A_list.append(self.A)





        #print("fe before a : {}".format(self.free_energy()), file=self.logfile)
        t0 = time.time()
        self.M_a(la_dep=la_dep)

        print("time M_a : {} s".format(time.time()-t0), file=self.logfile)
        #print("fe after a : {}".format(self.free_energy()), file=self.logfile)

        print("a = {}".format(self.a), file=self.logfile)
        print("", file=self.logfile)
        self.a_list.append(self.a)



        t0 = time.time()
        self.M_la()
        print("time M_la : {} s".format(time.time()-t0), file=self.logfile)

        #print("fe after la : {}".format(self.free_energy()))

        print("lambda = {}".format(self.la), file=self.logfile)
        print("", file=self.logfile)
        self.la_list.append(self.la)

        #t0 = time.time()
        self.M_sigma()
        #print("time M_sigma : {} s".format(time.time()-t0))

        print("sigma_2 = {}".format(self.sigma2), file=self.logfile)
        self.sigma2_list.append(self.sigma2)


        # E step
        #t0 = time.time()
        if E == "slow":
            self.E()
        elif E == "kalman":
            self.E_Kalman()
        else:
            raise Exception('E should be slow or kalman')
        print("time E : {} s".format(time.time()-t0), file=self.logfile)

        # FE history
        print("", file=self.logfile)
        self.FE_list.append(self.free_energy())
        print("last free energy : {}".format(self.FE_list[-1]), file=self.logfile)


def save_EM(algo, filename):
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
