import numpy as np
from scipy.special import logsumexp

def psudo_div(a, b):
    #zeros = np.zeros_like(a)
    zeros = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return zeros
def psudo_div_r(a,b):
    return a/b if b!=0 else 1

def normlize_col(arr):
    if arr.shape[0] ==1:
        return arr
    max = np.max(arr, axis=0)
    return arr-max

class GMM:
    def __init__(self, k, max_iter, tol=1e-5, S_scale = 1, name=None, using_S_thr=False, S_thr=0):
        self.name = name
        self.k = k
        _un_norm_Pi = np.random.random(self.k)
        self.Pi = _un_norm_Pi/np.sum(_un_norm_Pi)
        self.max_iter = max_iter
        self.r = np.zeros(self.k)
        self.l = np.array([])
        self.tol = tol
        self.S = None
        self.S_mask = None
        self.n_training_points = None
        self.using_S_thr = using_S_thr
        if self.using_S_thr:
            self.S_thr = S_thr
        self.S_scale = S_scale
        self.Mu = None
    def fit(self, x):
        n_data, dim_data = x.shape
        self.S = self.S_scale*np.ones((self.k, dim_data))
        self.Mu = np.max(x, axis=0)*np.random.random((self.k, dim_data)) # means(centroids of clusters)
        log_R_ik = np.zeros((self.k, n_data)) # log responsibility of each data point
        for iter in range(self.max_iter):
            for k in range(self.k):
                div_sqr = (x-self.Mu[k])**2
                index = np.sum(np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0), 1)
                #if float("inf") in index:
                #    print("a")
                index = index - np.min(index)
                prex_divd = np.sum(np.log(2*np.pi*self.S[k], out=np.zeros_like(self.S[k]), where=self.S[k] != 0))
                log_R_ik[k] = np.log(self.Pi[k], out=np.zeros_like(self.Pi[k]), where=self.Pi[k] != 0)-0.5*prex_divd-0.5*index

            R_ik = np.exp(log_R_ik)
            #if float("inf") in R_ik:
            #    print("a")
            R_i = np.sum(R_ik, axis=0)
            self.l = np.append(self.l, -np.sum(np.log(R_i, out=np.zeros_like(R_i), where=R_i != 0)))
            log_R_ik = normlize_col(log_R_ik)
            R_ik = np.exp(log_R_ik)
            R_i = np.sum(R_ik, axis=0)
            R_ik = psudo_div(R_ik, R_i)
            #R_i = np.sum(R_ik, axis=0)

            if iter > 1 and np.abs(self.l[iter]-self.l[iter-1]) <= self.tol*np.abs(self.l[iter]):
                break
            for k in range(self.k):
                self.r[k] = np.sum(R_ik[k])
                self.Pi[k] = self.r[k] / n_data
                un_norm_mu = np.matmul(R_ik[k], x)
                self.Mu[k] = np.divide(un_norm_mu, self.r[k], out=np.zeros_like(un_norm_mu), where=self.r[k] != 0)
                un_norm_s = np.sum(np.expand_dims(R_ik[k], axis=0).T * (x - self.Mu[k]) ** 2, 0)
                self.S[k] = np.divide(un_norm_s, self.r[k], out=np.zeros_like(un_norm_s), where=self.r[k] != 0)
            if self.using_S_thr:
                self.S[self.S < self.S_thr] = 0
        return self.l
    def predict(self, x):
        n_data, dim_data = x.shape
        log_R_ik = np.zeros((self.k, n_data))
        for k in range(self.k):
            div_sqr = (x - self.Mu[k]) ** 2
            index = np.sum(np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0), 1)
            prex_divd = np.sum(np.log(self.S[k], out=np.zeros_like(self.S[k]), where=self.S[k] != 0))
            log_R_ik[k] = np.log(self.Pi[k], out=np.zeros_like(self.Pi[k]), where=self.Pi[k] != 0) - 0.5 * prex_divd - 0.5 * index
        #R_ik = np.exp(np.clip(log_R_ik, -float("inf"), 0))
        return log_R_ik

    def loss(self, x):
        n_data, dim_data = x.shape
        log_R_ik = np.zeros((self.k, n_data)) # log responsibility of each data point
        for k in range(self.k):
            div_sqr = (x - self.Mu[k]) ** 2
            index = np.sum(np.divide(div_sqr, self.S[k], out=np.zeros_like(div_sqr), where=self.S[k] != 0), 1)
            # if float("inf") in index:
            #    print("a")
            index = index - np.min(index)
            prex_divd = np.sum(np.log(self.S[k], out=np.zeros_like(self.S[k]), where=self.S[k] != 0))
            log_R_ik[k] = np.log(self.Pi[k], out=np.zeros_like(self.Pi[k]),
                                 where=self.Pi[k] != 0) - 0.5 * prex_divd - 0.5 * index
        R_ik = np.exp(log_R_ik)
        R_i = np.sum(R_ik, axis=0)
        return -np.sum(np.log(R_i, out=np.zeros_like(R_i), where=R_i != 0))

